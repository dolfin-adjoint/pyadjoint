import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.testing import assert_allclose


def test_maximize():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    J = assemble(-(f-pi)**2*dx)
    Jhat = ReducedFunctional(J, Control(f))
    opt = maximize(Jhat)
    assert_allclose(opt.vector().get_local(), pi, rtol=1e-2)
