import pytest
import math
import numpy as np
from numpy.testing import assert_approx_equal
from numpy.random import rand
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


@pytest.mark.parametrize("exp", (exp, lambda x: 1 + np.expm1(x)))
def test_float_logexp(exp):
    a = AdjFloat(3.0)
    b = exp(a)
    c = log(b)
    assert_approx_equal(c, 3.0)

    b = log(a)
    c = exp(b)
    assert c, 3.0

    rf = ReducedFunctional(c, Control(a))
    assert_approx_equal(rf(a), 3.0)
    assert_approx_equal(rf(AdjFloat(1.0)), 1.0)
    assert_approx_equal(rf(AdjFloat(9.0)), 9.0)

    assert_approx_equal(rf.derivative(), 1.0)

    a = AdjFloat(3.0)
    b = exp(a)
    rf = ReducedFunctional(b, Control(a))
    assert_approx_equal(rf.derivative(), math.exp(3.0))

    a = AdjFloat(2.0)
    b = log(a)
    rf = ReducedFunctional(b, Control(a))
    assert_approx_equal(rf.derivative(), 1./2.)


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
    assert_approx_equal(rf.derivative(), 4.0 * (math.log(2.0)+1.0))

    # TODO: __rpow__ is not yet implemented


@pytest.mark.parametrize("B", [3,4])
@pytest.mark.parametrize("E", [6,5])
@pytest.mark.parametrize("f", ["b**e", "e**b"])
def test_pow_hessian(B, E, f):
    # Testing issue 126
    set_working_tape(Tape())
    e = AdjFloat(E)
    b = AdjFloat(B)
    f = eval(f)
    J = ReducedFunctional(f, Control(e))
    results = taylor_to_dict(J, e, AdjFloat(10))
    for (i, Ri) in enumerate(["R0","R1","R2"]):
        assert(min(results[Ri]["Rate"]) >= i + 0.95)


def test_min_max():
    from pyadjoint.adjfloat import min, max
    set_working_tape(Tape())
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    controls = [Control(a), Control(b)]

    J = 5*max(a, b)**2 + 3*min(a, b)**3
    Ja = 5*a**2 + 3*b**3
    Jb = 5*b**2 + 3*a**3
    Jhat = ReducedFunctional(J, controls)
    Jhat_a = ReducedFunctional(Ja, controls)
    Jhat_b = ReducedFunctional(Jb, controls)

    assert Jhat([3.0, 2.0]) == J
    dJ = Jhat.derivative()
    assert dJ[0] == 30.
    assert dJ[1] == 36.
    assert Jhat([2.0, 3.0]) == J
    dJ = Jhat.derivative()
    assert dJ[0] == 36.
    assert dJ[1] == 30.

    x = list(rand(2))
    h = list(rand(2))
    if x[0] > x[1]:
        other = Jhat_a
    else:
        other = Jhat_b
    assert Jhat(x) == other(x)
    a, b = Jhat.derivative()
    a2, b2 = Jhat.hessian(h)
    oa, ob = other.derivative()
    oa2, ob2 = other.hessian(h)
    assert a == oa
    assert b == ob
    assert a2 == oa2
    assert b2 == ob2

    a, b = 3., 3.
    x = [a, b]
    assert Jhat(x) == Jhat_a(x)
    a, b = Jhat.derivative()
    a2, b2 = Jhat.hessian(h)
    oa, ob = other.derivative()
    oa2, ob2 = other.hessian(h)
    # For min/max of (a, b) with a == b, we return a.
    assert a == oa + ob
    assert b == 0.
    assert a2 == oa2 + h[0]*ob2/h[1]
    assert b2 == 0.


def test_float_components():
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    c = a * b
    assert c == 6.0
    rf = ReducedFunctional(c, (Control(a), Control(b)),
                           derivative_components=(1,))
    assert rf((a, b)) == 6.0
    assert rf((AdjFloat(1.0), AdjFloat(2.0))) == 2.0
    assert rf((AdjFloat(3.0), AdjFloat(2.0))) == 6.0
    assert rf.derivative() == [0.0, 3.0]
    assert rf((AdjFloat(4.0), AdjFloat(3.0))) == 12.0
    assert rf.derivative() == [0.0, 4.0]

def test_float_components_minimize():
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    c = (a + b)**3
    assert(c == 125.0)
    rf = ReducedFunctional(c, (Control(a), Control(b)),
                           derivative_components=(0,))
    z = minimize(rf)
    # this should minimise c over a, leaving b fixed
    # check that b is fixed
    assert(z[1] == 2.0)
    # check that a + b = 0 (the minimum)
    assert(abs(z[0] + z[1]) < 5.0e-3)

    # check that we can stop annotating, change
    # the values of the inputs, and minimise still
    # keeping b fixed (to the new value)
    with stop_annotating():
        rf((AdjFloat(1.0), AdjFloat(1.0)))
        z = minimize(rf)
        assert(z[1] == 1.0)
        assert(abs(z[0] + z[1]) < 5.0e-3)


def test_scipy_failure():
    a = AdjFloat(math.nan)
    b = 2 * a
    rf = ReducedFunctional(b, Control(a))
    with pytest.raises(SciPyConvergenceError):
        _ = minimize(rf)
