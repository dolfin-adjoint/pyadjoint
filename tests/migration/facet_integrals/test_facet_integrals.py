import pytest
pytest.importorskip("fenics")
pytest.importorskip("fenics_adjoint")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def test_facet_integrals():
    mesh = UnitIntervalMesh(2)
    W = FunctionSpace(mesh, "CG", 1)

    u = Function(W)
    g = project(Constant(1), W)
    u_ = TrialFunction(W)
    v = TestFunction(W)

    F = u_*v * dx - g*v*dx
    solve(lhs(F) == rhs(F), u)

    J = assemble(0.5 * inner(u("+"), u("+")) * dS)

    # Reduced functional with single control
    m = Control(g)

    Jhat = ReducedFunctional(J, m)
    h = Function(W)
    h.vector()[:] = rand(W.dim())
    assert taylor_test(Jhat, g, h) > 1.9
