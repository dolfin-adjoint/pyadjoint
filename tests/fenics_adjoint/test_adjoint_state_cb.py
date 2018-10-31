import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *


def test_time_dependent():
    mesh = UnitSquareMesh(5, 5)
    V = FunctionSpace(mesh, "CG", 1)

    U_prev = project(Expression("sin(x[0])", degree=1), V)
    control = Control(U_prev)
    U = Function(V)

    u = TrialFunction(V)
    v = TestFunction(V)

    dt = 0.1
    T = 1.0

    a = inner(u, v)*dx + dt * inner(grad(u), grad(v))*dx
    L = inner(U_prev, v)*dx

    adj_states = []
    adj_cb = lambda adj_sol: adj_states.append(adj_sol)

    t = 0
    while t < T:
        solve(a == L, U, adj_cb=adj_cb)
        t += dt
        U_prev.assign(U)

    J = assemble(inner(U, U)*dx)
    Jhat = ReducedFunctional(J, control)
    dJdm = Jhat.derivative()

    dFdm = -inner(u, v)*dx
    adj_int = adj_states[-1]
    dJdm_manual = -assemble(action(dFdm, adj_int))
    assert abs(dJdm.vector().get_local() - dJdm_manual.get_local()).sum() <= 0

