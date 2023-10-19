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
end = 0.4
timestep = Constant(1.0/n)
steps = int(end/float(timestep))

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def J(ic, solve_type, checkpointing):
    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)
    u_.assign(ic)
    nu = Constant(0.0001)
    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    if solve_type == "NLVS":
        problem = NonlinearVariationalProblem(F, u, bcs=bc)
        solver = NonlinearVariationalSolver(problem)

    def time_advance():
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)
    tape = get_working_tape()
    t += float(timestep)
    if checkpointing:
        for t in tape.timestepper(np.arange(t, end, float(timestep))):
            time_advance()
    else:
        while (t <= end):
            time_advance()
            t += float(timestep)

    return assemble(u_*u_*dx + ic*ic*dx), u_


@pytest.mark.parametrize("solve_type, checkpointing",
                         [("solve", True),
                          ("NLVS", True),
                          ("solve", False),
                          ("NLVS", False),
                          ])
def test_burgers_newton(solve_type, checkpointing):
    """Adjoint-based gradient tests with and without checkpointing.
    """
    if checkpointing:
        tape = get_working_tape()
        tape.progress_bar = ProgressBar
        tape.enable_checkpointing(Revolve(steps, steps//3))
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2.*pi*x), V)
    val, _ = J(ic, solve_type, checkpointing)
    if checkpointing:
        assert len(tape.timesteps) == steps

    # Test recompute
    Jhat = ReducedFunctional(val, Control(ic))
    val1 = Jhat(ic)
    assert(np.allclose(val1, val))

    # Taylor test
    h = Function(V)
    h.assign(1, annotate=False)
    assert taylor_test(Jhat, ic, h) > 1.9


@pytest.mark.parametrize("solve_type",
                         ["solve", "NLVS"])
def test_checkpointing_validity(solve_type):
    """Compare forward and backward results with and without checkpointing.
    """
    # Without checkpointing
    tape = get_working_tape()
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2.*pi*x), V)
    val0, u0 = J(ic, solve_type, False)
    Jhat = ReducedFunctional(val0, Control(ic))
    dJ0 = Jhat.derivative()
    tape.clear_tape()

    # With checkpointing
    tape.progress_bar = ProgressBar
    tape.enable_checkpointing(Revolve(steps, 10))
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2.*pi*x), V)
    val1, u1 = J(ic, solve_type, True)
    Jhat = ReducedFunctional(val1, Control(ic))
    dJ1 = Jhat.derivative()
    print(val0, val1)
    assert len(tape.timesteps) == steps
    assert np.allclose(val0, val1, rtol=1e-2)
    assert np.allclose(u0.dat.data_ro[:], u1.dat.data_ro[:])
    assert np.allclose(dJ0.dat.data_ro[:], dJ1.dat.data_ro[:])
