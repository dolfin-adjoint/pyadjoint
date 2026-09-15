"""Accumulation of adjoint/tlm/hessian contributions in `BlockVariable`.

A block variable collects a contribution per fan-in edge, so these paths run
whenever a variable is used more than once. Every failure mode here is a
silently wrong derivative rather than an exception, so the behaviour is pinned
explicitly.
"""
import numpy as np
import pytest

import pyadjoint.block_variable as block_variable_module
from pyadjoint import (
    AdjFloat,
    Control,
    ReducedFunctional,
    Tape,
    continue_annotation,
    create_overloaded_object,
    set_working_tape,
)
from pyadjoint.block_variable import BlockVariable, _accumulate
from pyadjoint.drivers import compute_derivative, compute_hessian, compute_tlm
from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.tape import get_working_tape


@pytest.fixture(autouse=True)
def _tape():
    continue_annotation()
    set_working_tape(Tape())
    yield
    set_working_tape(Tape())


class Mutable(OverloadedType):
    """Accumulates in place and returns `self`, as a mutable backend type does."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def _ad_iadd(self, other):
        self.value += other.value
        return self


class MutableReturningNone(Mutable):
    """Honours the pre-2026 contract, which documented ``Returns: None``."""

    def _ad_iadd(self, other):
        self.value += other.value


class NoAdIadd:
    """A block may return a bare value; `numpy_adjoint` returns an ndarray."""

    def __init__(self, value):
        self.value = value

    def __iadd__(self, other):
        self.value += other.value
        return self


# -- the accumulation primitive -------------------------------------------


def test_immutable_value_is_rebound():
    """An immutable type cannot accumulate in place, so the sum is returned."""
    assert _accumulate(AdjFloat(1.0), AdjFloat(2.0)) == 3.0


def test_mutable_value_accumulates_in_place():
    current = Mutable(1.0)
    assert _accumulate(current, Mutable(2.0)) is current
    assert current.value == 3.0


def test_ad_iadd_returning_none_does_not_discard_the_value():
    """`None` means "mutated in place", not "the value is now nothing".

    An override written against the old contract would otherwise reset the
    block variable to `None`, after which the next contribution overwrites
    instead of accumulating and the derivative silently comes back zero.
    """
    current = MutableReturningNone(1.0)
    assert _accumulate(current, MutableReturningNone(2.0)) is current
    assert current.value == 3.0


def test_value_without_ad_iadd_falls_back_to_in_place_add():
    current = NoAdIadd(1.0)
    assert _accumulate(current, NoAdIadd(2.0)) is current
    assert current.value == 3.0


def test_ndarray_accumulates_in_place_and_keeps_dtype():
    """`numpy_adjoint` hands back a bare ndarray; it must not be reallocated.

    Accumulating out of place would also promote under NEP 50, so the
    derivative would come back with a different dtype than the control.
    """
    current = np.zeros(3, dtype=np.float32)
    result = _accumulate(current, np.float64(1.0))
    assert result is current
    assert result.dtype == np.float32
    assert np.array_equal(result, np.ones(3, dtype=np.float32))


def test_numpy_slice_fan_in_takes_the_fallback():
    """The `+` fallback is load-bearing, not dead code.

    Casting the `AdjFloatExprBlock` returns to `AdjFloat` makes *that* block
    hand back an overloaded value, but says nothing about any other block.
    `NumpyArraySliceBlock.evaluate_adj_component` returns a bare
    `numpy.zeros(...)`, so an array sliced more than once accumulates through
    the fallback. Asserted here so the fallback is not removed as unreachable.
    """
    import numpy_adjoint  # noqa: F401  (registers the ndarray overload)

    dispatched_via = []
    original = block_variable_module._accumulate

    def record(current, val):
        dispatched_via.append(getattr(current, "_ad_iadd", None) is None)
        return original(current, val)

    block_variable_module._accumulate = record
    try:
        a = create_overloaded_object(np.array([-2.0, 3.0]))
        rf = ReducedFunctional(a[0] * a[0] + a[1], Control(a))
        derivative = np.asarray(rf.derivative())
    finally:
        block_variable_module._accumulate = original

    assert np.allclose(derivative, [-4.0, 1.0])
    assert dispatched_via and all(dispatched_via), (
        "expected every accumulation to take the '+' fallback, got "
        f"{dispatched_via}"
    )


# -- the three block-variable slots ---------------------------------------


@pytest.mark.parametrize("add_output, slot", [
    (BlockVariable.add_adj_output, "adj_value"),
    (BlockVariable.add_tlm_output, "tlm_value"),
    (BlockVariable.add_hessian_output, "hessian_value"),
])
def test_slot_accumulates_an_immutable_value(add_output, slot):
    """All three slots must handle a value type that rebinds rather than mutates."""
    bv = BlockVariable(None)
    add_output(bv, AdjFloat(1.0))
    add_output(bv, AdjFloat(2.0))
    add_output(bv, AdjFloat(4.0))
    assert getattr(bv, slot) == 7.0


@pytest.mark.parametrize("add_output, slot", [
    (BlockVariable.add_adj_output, "adj_value"),
    (BlockVariable.add_tlm_output, "tlm_value"),
    (BlockVariable.add_hessian_output, "hessian_value"),
])
def test_slot_accumulates_a_type_defining_only_ad_iadd(add_output, slot):
    """A backend may define `_ad_iadd` without an `__iadd__` operator.

    This is the `dolfinx_adjoint` case: `Function` is a `ufl.Coefficient`, so
    `+=` would build a symbolic sum instead of adding dof values.
    """
    bv = BlockVariable(None)
    add_output(bv, Mutable(1.0))
    add_output(bv, Mutable(2.0))
    assert isinstance(getattr(bv, slot), Mutable)
    assert getattr(bv, slot).value == 3.0


# -- end to end ------------------------------------------------------------


@pytest.mark.parametrize("build, derivative, second_derivative", [
    (lambda a: a * 3.0, 3.0, 0.0),
    (lambda a: a + a, 2.0, 0.0),
    (lambda a: a - a, 0.0, 0.0),
    (lambda a: a * a + a, 5.0, 2.0),
    (lambda a: a * a * a, 12.0, 12.0),
])
def test_repeated_dependency_accumulates(build, derivative, second_derivative):
    """A variable used twice in one expression fans in to one block variable.

    `a + a` is the minimal case: the local derivative is the constant 1, so
    neither factor of ``codegen(...) * adj_input`` is overloaded.
    """
    a = AdjFloat(2.0)
    J = build(a)
    control = Control(a)
    assert float(compute_derivative(J, control)) == pytest.approx(derivative)
    assert float(compute_tlm(J, control, AdjFloat(1.0))) == pytest.approx(derivative)
    assert float(compute_hessian(J, control, AdjFloat(1.0))) == pytest.approx(second_derivative)


def test_accumulation_does_not_extend_the_tape():
    """`_ad_iadd` runs during the reverse sweep and must not annotate.

    `AdjFloat.__add__` is annotated, so an unguarded `self += other` would add
    a block per accumulation and corrupt any later recompute.
    """
    tape = get_working_tape()
    a = AdjFloat(2.0)
    J = a * a + a
    before = len(tape.get_blocks())
    compute_derivative(J, Control(a))
    assert len(tape.get_blocks()) == before
