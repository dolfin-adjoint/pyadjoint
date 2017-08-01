import pytest

# this test only uses AdjFloat so it should run under both firedrake and fenics
try:
    from fenics_adjoint import *
except ImportError:
    from firedrake_adjoint import *

def test_float_addition():
    a = AdjFloat(1.0)
    b = AdjFloat(2.0)
    c = a + b
    assert c == 3.0
    rf = ReducedFunctional(c, a)
    assert rf(a) == 3.0
    assert rf(AdjFloat(1.0)) == 3.0
    assert rf(AdjFloat(3.0)) == 5.0

    d = a + a
    assert d == 2.0
    rf = ReducedFunctional(d, a)
    assert rf(AdjFloat(1.0)) == 2.0
    assert rf(AdjFloat(2.0)) == 4.0

    # this tests AdjFloat.__radd__
    e = 3.0 + a
    assert e == 4.0
    rf = ReducedFunctional(e, a)
    assert rf(a) == 4.0
    assert rf(AdjFloat(7.0)) == 10.0

def test_float_subtraction():
    a = AdjFloat(1.0)
    b = AdjFloat(2.0)
    c = a - b
    assert c == -1.0
    d = a - a
    assert d == 0.0
    rf = ReducedFunctional(c, a)
    assert rf(a) == -1.0
    assert rf(AdjFloat(1.0)) == -1.0
    assert rf(AdjFloat(3.0)) == 1.0
    rf = ReducedFunctional(d, a)
    assert rf(AdjFloat(1.0)) == 0.0
    assert rf(AdjFloat(2.0)) == 0.0

    # this tests AdjFloat.__rsub__
    e = 3.0 - a
    e.get_block_output()
    assert e == 2.0
    rf = ReducedFunctional(e, a)
    assert rf(b) == 1.0
    assert rf(AdjFloat(3.0)) == 0.0

def test_float_multiplication():
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    c = a * b
    assert c == 6.0
    d = a * a
    assert d == 9.0
    rf = ReducedFunctional(c, a)
    assert rf(a) == 6.0
    assert rf(AdjFloat(1.0)) == 2.0
    assert rf(AdjFloat(3.0)) == 6.0
    rf = ReducedFunctional(d, a)
    assert rf(AdjFloat(1.0)) == 1.0

    # this tests AdjFloat.__rmul__
    e = 5.0 * a
    e.get_block_output()
    assert e == 15.0
    rf = ReducedFunctional(e, a)
    assert rf(b) == 10.0
    assert rf(AdjFloat(3.0)) == 15.0
    assert rf(AdjFloat(2.0)) == 10.0

def test_float_exponentiation():
    a = AdjFloat(3.0)
    b = AdjFloat(2.0)
    c = a ** b
    assert c == 9.0
    d = a ** a
    assert d == 27.0
    rf = ReducedFunctional(c, a)
    assert rf(a) == 9.0
    assert rf(AdjFloat(1.0)) == 1.0
    assert rf(AdjFloat(3.0)) == 9.0
    rf = ReducedFunctional(d, a)
    assert rf(AdjFloat(1.0)) == 1.0
    assert rf(AdjFloat(2.0)) == 4.0

    # TODO: __rpow__ is not yet implemented
