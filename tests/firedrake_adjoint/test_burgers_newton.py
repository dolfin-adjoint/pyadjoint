"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import Revolve
import numpy as np

set_log_level(CRITICAL)
continue_annotation()
n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)
end = 0.5
timestep = Constant(1.0/n)
steps = int(end/float(timestep)) + 1
def Dt(u, u_, timestep):
    return (u - u_)/timestep


def J(ic, solve_type, checkpointing):
    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

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
    tape = get_working_tape()
    tape.end_timestep()
    t += float(timestep)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    
    def time_advance():
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)

    if checkpointing:
        for t in tape.timestepper(np.arange(t, end, float(timestep))):
            time_advance()
    else:
        while (t <= end):
            time_advance()
            t += float(timestep)

    return assemble(u_*u_*dx + ic*ic*dx)


@pytest.mark.parametrize("solve_type, snaps_in_ram, checkpointing",
                         [("solve", steps//3, True),
                          ("solve", steps//2, True),
                          ("solve", 1, True),
                          ("NLVS", steps//3, True),
                          ("NLVS", steps//2, True),
                          ("NLVS", 1, True),
                          ("solve", steps, True),
                          ("NLVS", steps, True),
                          ("solve", steps, False),
                          ("NLVS", steps, False),
                        ])
def test_burgers_newton(solve_type, snaps_in_ram, checkpointing):
    if checkpointing:
        tape = get_working_tape()
        tape.progress_bar = ProgressBar
        tape.enable_checkpointing(Revolve(steps, snaps_in_ram))
    print(snaps_in_ram)
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2.*pi*x), V)

    val = J(ic, solve_type, checkpointing)
    if checkpointing:
        assert len(tape.timesteps) == steps
    Jhat = ReducedFunctional(val, Control(ic))
    dJ = Jhat.derivative()
    val1 = Jhat(ic)
    assert(np.allclose(val1, val))
    dJbar = Jhat.derivative()
    assert np.allclose(dJ.dat.data_ro[:], dJbar.dat.data_ro[:])
    h = Function(V)
    h.assign(1, annotate=False)
    assert taylor_test(Jhat, ic, h) > 1.9

test_burgers_newton("solve", 1, True)