try:
    from fenics_adjoint import *
except ImportError:
    from firedrake_adjoint import *


def test_compute_gradient():
    a = AdjFloat(1.0)
    b = AdjFloat(2.0)

    J = a*b
    c = Control(a)
    assert isinstance(compute_gradient(J, c), AdjFloat)
    assert isinstance(compute_gradient(J, [c]), list)


def test_reduced_functional():
    a = AdjFloat(1.0)
    b = AdjFloat(2.0)

    J = a*b
    c = Control(a)
    Jhat = ReducedFunctional(J, c)
    assert isinstance(Jhat.derivative(), AdjFloat)

    Jhat = ReducedFunctional(J, [c])
    assert isinstance(Jhat.derivative(), list)

