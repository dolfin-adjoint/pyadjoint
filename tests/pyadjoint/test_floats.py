import pytest
from math import log
from numpy.testing import assert_approx_equal
from pyadjoint import *


def test_float_addition():
    a = AdjFloat(1.0)
    b = AdjFloat(2.0)
    c = a + b
    assert c == 3.0
    rf = ReducedFunctional(c, Control(a))
    assert rf(a) == 3.0
    assert rf(AdjFloat(1.0)) == 3.0
    assert rf(AdjFloat(3.0)) == 5.0
    assert rf.derivative() == 1.0

    d = a + a
    assert d == 2.0
    rf = ReducedFunctional(d, Control(a))
    assert rf(AdjFloat(1.0)) == 2.0
    assert rf(AdjFloat(2.0)) == 4.0
    assert rf.derivative() == 2.0

    # this tests AdjFloat.__radd__
    e = 3.0 + a
    assert e == 4.0
    rf = ReducedFunctional(e, Control(a))
    assert rf(a) == 4.0
    assert rf(AdjFloat(7.0)) == 10.0
    assert rf.derivative() == 1.0


def test_float_subtraction():
    a = AdjFloat(1.0)
    b = AdjFloat(2.0)
    c = a - b
    assert c == -1.0
    d = a - a
    assert d == 0.0
    rf = ReducedFunctional(c, Control(a))
    assert rf(a) == -1.0
    assert rf(AdjFloat(1.0)) == -1.0
    assert rf(AdjFloat(3.0)) == 1.0
    assert rf.derivative() == 1.0
    rf = ReducedFunctional(d, Control(a))
    assert rf(AdjFloat(1.0)) == 0.0
    assert rf(AdjFloat(2.0)) == 0.0
    assert rf.derivative() == 0.0

    # this tests AdjFloat.__rsub__
    e = 3.0 - a
    e.block_variable
    assert e == 2.0
    rf = ReducedFunctional(e, Control(a))
    assert rf(b) == 1.0
    assert rf(AdjFloat(3.0)) == 0.0
    assert rf.derivative() == -1.0


def test_float_multiplication():
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    c = a * b
    assert c == 6.0
    d = a * a
    assert d == 9.0
    rf = ReducedFunctional(c, Control(a))
    assert rf(a) == 6.0
    assert rf(AdjFloat(1.0)) == 2.0
    assert rf(AdjFloat(3.0)) == 6.0
    assert rf.derivative() == 2.0
    rf = ReducedFunctional(d, Control(a))
    assert rf(AdjFloat(1.0)) == 1.0
    assert rf.derivative() == 2.0
    assert rf(AdjFloat(5.0)) == 25.0
    assert rf.derivative() == 10.0

    # this tests AdjFloat.__rmul__
    e = 5.0 * a
    e.block_variable
    assert e == 15.0
    rf = ReducedFunctional(e, Control(a))
    assert rf(b) == 10.0
    assert rf(AdjFloat(3.0)) == 15.0
    assert rf(AdjFloat(2.0)) == 10.0
    assert rf.derivative() == 5.0


def test_float_division():
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    c = a / b
    assert c == 1.5
    d = a / a
    assert d == 1.0
    rf = ReducedFunctional(c, Control(a))
    assert rf(a) == 1.5
    assert rf(AdjFloat(1.0)) == 0.5
    assert rf(AdjFloat(3.0)) == 1.5
    assert rf.derivative() == 0.5
    rf = ReducedFunctional(c, Control(b))
    assert rf.derivative() == -0.75
    assert rf(b) == 1.5
    assert rf(AdjFloat(1.0)) == 3.0
    assert rf(AdjFloat(3.0)) == 1.0
    assert_approx_equal(rf.derivative(), -3.0/9.0)  # b is now 3.0
    rf = ReducedFunctional(d, Control(a))
    assert rf(AdjFloat(1.0)) == 1.0
    assert rf.derivative() == 0.0
    assert rf(AdjFloat(5.0)) == 1.0
    assert rf.derivative() == 0.0

    # this tests AdjFloat.__rmul__
    e = 5.0 * a
    e.block_variable
    assert e == 15.0
    rf = ReducedFunctional(e, Control(a))
    assert rf(b) == 10.0
    assert rf(AdjFloat(3.0)) == 15.0
    assert rf(AdjFloat(2.0)) == 10.0
    assert rf.derivative() == 5.0


def test_float_neg():
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    c = - a
    d = - a * b
    assert c == - 3.0
    assert d == - 6.0
    rf1 = ReducedFunctional(c, Control(a))
    assert rf1(a) == - 3.0
    assert rf1.derivative() == - 1.0
    rf2 = ReducedFunctional(d, Control(a))
    assert rf2(a) == - 6.0
    assert rf2.derivative() == - 2.0
    assert rf2(b) == - 4.0
    assert rf2.derivative() == - 2.0
    e = - AdjFloat(4.0)
    f = AdjFloat(-5.0)
    g = - AdjFloat(-7.0)
    assert rf1(e) == 4.0
    assert rf1.derivative() == - 1.0
    assert rf1(f) == 5.0
    assert rf1.derivative() == - 1.0
    assert rf1(g) == - 7.0
    assert rf1.derivative() == - 1.0
    assert rf2(e) == 8.0
    assert rf2.derivative() == - 2.0
    assert rf2(f) == 10.0
    assert rf2.derivative() == - 2.0
    assert rf2(g) == - 14.0
    assert rf2.derivative() == - 2.0


def test_float_exponentiation():
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    c = a ** b
    assert c == 9.0
    d = a ** a
    assert d == 27.0
    rf = ReducedFunctional(c, Control(a))
    assert rf(a) == 9.0
    assert rf(AdjFloat(1.0)) == 1.0
    assert rf(AdjFloat(3.0)) == 9.0
    # d(a**b)/da = b*a**(b-1)
    assert rf.derivative() == 6.0
    rf = ReducedFunctional(d, Control(a))
    assert rf(AdjFloat(1.0)) == 1.0
    assert rf(AdjFloat(2.0)) == 4.0
    # d(a**a)/da = dexp(a log(a))/da = a**a * (log(a) + 1)
    assert_approx_equal(rf.derivative(), 4.0 * (log(2.0)+1.0))

    # TODO: __rpow__ is not yet implemented


@pytest.mark.parametrize("B", range(2,5))
@pytest.mark.parametrize("E", [-2,-1,2,3])
def test_pow_hessian(B, E):
    # Testing issue 126
    set_working_tape(Tape())
    e = AdjFloat(E)
    b = AdjFloat(B)
    f = b**e
    J = ReducedFunctional(f, Control(e))
    results = taylor_to_dict(J, e, AdjFloat(1))
    for (i, Ri) in enumerate(["R0","R1","R2"]):
        assert(min(results[Ri]["Rate"]) >= i + 0.95)
