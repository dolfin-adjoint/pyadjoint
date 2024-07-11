from contextlib import contextmanager

import pytest

from pyadjoint import *


@pytest.fixture(autouse=True)
def _():
    get_working_tape().clear_tape()
    continue_annotation()
    continue_reverse_over_forward()
    yield
    get_working_tape().clear_tape()
    pause_annotation()
    pause_reverse_over_forward()


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
def test_exp(a_val, tlm_a_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    x = exp(a)
    _ = compute_gradient(x.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert adj_value == exp(a_val) * tlm_a_val


@pytest.mark.parametrize("a_val", [2.0, 3.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
def test_log(a_val, tlm_a_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    x = log(a)
    _ = compute_gradient(x.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert adj_value == -tlm_a_val / (a_val ** 2)


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5, None])
@pytest.mark.parametrize("b_val", [4.25, -4.25])
@pytest.mark.parametrize("tlm_b_val", [5.8125, -5.8125, None])
def test_add(a_val, tlm_a_val, b_val, tlm_b_val):
    a = AdjFloat(a_val)
    if tlm_a_val is not None:
        a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(b_val)
    if tlm_b_val is not None:
        b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = a + b
    if tlm_a_val is None and tlm_b_val is None:
        assert x.block_variable.tlm_value is None
    else:
        assert (x.block_variable.tlm_value ==
                (0.0 if tlm_a_val is None else tlm_a_val) + (0.0 if tlm_b_val is None else tlm_b_val))


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5, None])
@pytest.mark.parametrize("b_val", [4.25, -4.25])
@pytest.mark.parametrize("tlm_b_val", [5.8125, -5.8125, None])
def test_sub(a_val, tlm_a_val, b_val, tlm_b_val):
    a = AdjFloat(a_val)
    if tlm_a_val is not None:
        a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(b_val)
    if tlm_b_val is not None:
        b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = a - b
    if tlm_a_val is None and tlm_b_val is None:
        assert x.block_variable.tlm_value is None
    else:
        assert (x.block_variable.tlm_value ==
                (0.0 if tlm_a_val is None else tlm_a_val) - (0.0 if tlm_b_val is None else tlm_b_val))


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
def test_neg(a_val, tlm_a_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    x = -a
    assert x.block_variable.tlm_value == -tlm_a_val
