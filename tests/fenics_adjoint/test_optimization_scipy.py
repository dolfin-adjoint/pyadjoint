import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.testing import assert_allclose, assert_approx_equal


def test_maximize():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    J = assemble(-(f-pi)**2*dx)
    Jhat = ReducedFunctional(J, Control(f))
    opt = maximize(Jhat)
    assert_allclose(opt.vector().get_local(), pi, rtol=1e-2)

def test_adjfloat_minimize():
    x = AdjFloat(5)
    J = (x - 1)**2
    Jhat = ReducedFunctional(J, Control(x))

    # Test with a callback that modifies the tape values. This used to cause problems with an immutable Control.
    def callback(xk):
        new_J = Jhat(AdjFloat(2))

    opt = minimize(Jhat, callback=callback)
    assert_approx_equal(opt, 1)
