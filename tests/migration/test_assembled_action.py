import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def test_assembled_action():
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)

    data = Function(V, name="Data")
    data.vector()[0] = 1.0

    u = TrialFunction(V)
    v = TestFunction(V)
    mass = inner(u, v)*dx
    M = assemble(mass)

    rhs = M*data.vector()
    soln = Function(V)

    solve(M, soln.vector(), rhs)

    J = assemble(inner(soln, soln)*dx)
    control = Control(data)
    dJdic = compute_gradient(J, control)

    h = Function(V)
    h.vector()[:] = rand(V.dim())

    dJdm = dJdic._ad_dot(h)

    minconv = taylor_test(ReducedFunctional(J, control), data, h, dJdm=dJdm)
    assert minconv > 1.9