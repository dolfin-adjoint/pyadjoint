"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake_adjoint import *
from hrevolve import revolve
import numpy as np

set_log_level(CRITICAL)

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)


def Dt(u, u_, timestep):
    return (u - u_)/timestep


def J(ic, solve_type):
    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, ic, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    if solve_type == "NLVS":
        problem = NonlinearVariationalProblem(F, u, bcs=bc)
        solver = NonlinearVariationalSolver(problem)
        solver.solve()
    else:
        solve(F == 0, u, bc)
    u_.assign(u)
    t += float(timestep)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    end = 0.2
    tape = get_working_tape()
    for t in tape.timestepper(np.arange(t, end, float(timestep))):
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)

    return assemble(u_*u_*dx + ic*ic*dx)


@pytest.mark.parametrize("solve_type",
                         ["solve", "NLVS"])
def test_burgers_newton(solve_type):
    tape = get_working_tape()
    tape.enable_checkpointing(revolve(6, 2, uf=1, ub=1, print_table=False))

    x, = SpatialCoordinate(mesh)
    ic = project(sin(2.*pi*x), V)

    val = J(ic, solve_type)
    assert len(tape.timesteps) == 7
    breakpoint()
    Jhat = ReducedFunctional(val, Control(ic))
    dJ = Jhat.derivative()
    Jhat(ic)
    dJbar = Jhat.derivative()
    assert np.allclose(dJ.dat.data_ro[:], dJbar.dat.data_ro[:])

    h = Function(V)
    h.assign(1, annotate=False)
    assert taylor_test(Jhat, ic, h) > 1.9


test_burgers_newton("solve")
