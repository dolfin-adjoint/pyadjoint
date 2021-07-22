import pytest

pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def test_slice():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - f ** 2 * v * dx
    solve(F == 0, u, bc)

    vec = assemble(inner(grad(u), grad(v)) * dx, _ad_overload_vector=True)
    J = vec[0] + vec[3]
    Jhat = ReducedFunctional(J, Control(f))
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(Jhat, f, h) > 1.9
