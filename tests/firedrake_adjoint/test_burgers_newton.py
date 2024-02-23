"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import Revolve, SingleMemoryStorageSchedule, MixedCheckpointSchedule,\
    NoneCheckpointSchedule, StorageType
import numpy as np
set_log_level(CRITICAL)
continue_annotation()
n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)
end = 0.2
timestep = Constant(1.0/n)
steps = int(end/float(timestep)) + 1


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

    tape = get_working_tape()
    t += float(timestep)
    for t in tape.timestepper(np.arange(t, end + t, float(timestep))):
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)

    return assemble(u_*u_*dx + ic*ic*dx), u_


@pytest.mark.parametrize("solve_type, checkpointing",
                         [("solve", "Revolve"),
                          ("NLVS", "Revolve"),
                          ("solve", "SingleMemory"),
                          ("NLVS", "SingleMemory"),
                          ("solve", "NoneAdjoint"),
                          ("NLVS", "NoneAdjoint"),
                          ("solve", "Mixed"),
                          ("NLVS", "Mixed"),
                          ("solve", None),
                          ("NLVS", None),
                          ])
def test_burgers_newton(solve_type, checkpointing):
    """Adjoint-based gradient tests with and without checkpointing.
    """
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    if checkpointing:
        if checkpointing == "Revolve":
            schedule = Revolve(steps, steps//3)
        if checkpointing == "SingleMemory":
            schedule = SingleMemoryStorageSchedule()
        if checkpointing == "Mixed":
            schedule = MixedCheckpointSchedule(steps, steps//3, storage=StorageType.RAM)
        if checkpointing == "NoneAdjoint":
            schedule = NoneCheckpointSchedule()
        tape.enable_checkpointing(schedule)
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2.* pi * x), V)
    val, _ = J(ic, solve_type, checkpointing)
    if checkpointing:
        assert len(tape.timesteps) == steps
    # if checkpointing != "NoneAdjoint":
    #     Jhat = ReducedFunctional(val, Control(ic))
    # dJ = Jhat.derivative()

    # # Recomputing the functional with a modified control variable
    # # before the recompute test.
    # Jhat(project(sin(pi*x), V))

    # # Recompute test
    # assert(np.allclose(Jhat(ic), val))
    # if checkpointing != "NoneAdjoint":
    #     dJbar = Jhat.derivative()
    #     # Test recompute adjoint-based gradient
    #     assert np.allclose(dJ.dat.data_ro[:], dJbar.dat.data_ro[:])

    #     # Taylor test
    #     h = Function(V)
    #     h.assign(1, annotate=False)
    #     assert taylor_test(Jhat, ic, h) > 1.9


# @pytest.mark.parametrize("solve_type, checkpointing",
#                          [("solve", "Revolve"),
#                           ("NLVS", "Revolve")
#                           ])
# def test_checkpointing_validity(solve_type, checkpointing):
#     """Compare forward and backward results with and without checkpointing.
#     """
#     # Without checkpointing
#     tape = get_working_tape()
#     tape.progress_bar = ProgressBar
#     x, = SpatialCoordinate(mesh)
#     ic = project(sin(2.*pi*x), V)

#     val0, u0 = J(ic, solve_type, False)
#     Jhat = ReducedFunctional(val0, Control(ic))
#     dJ0 = Jhat.derivative()
#     tape.clear_tape()

#     # With checkpointing
#     tape.progress_bar = ProgressBar
#     if checkpointing == "Revolve":
#         tape.enable_checkpointing(Revolve(steps, steps//3))
#     x, = SpatialCoordinate(mesh)
#     ic = project(sin(2.*pi*x), V)
#     val1, u1 = J(ic, solve_type, True)
#     Jhat = ReducedFunctional(val1, Control(ic))
#     dJ1 = Jhat.derivative()
#     assert len(tape.timesteps) == steps
#     assert np.allclose(val0, val1)
#     assert np.allclose(u0.dat.data_ro[:], u1.dat.data_ro[:])
#     assert np.allclose(dJ0.dat.data_ro[:], dJ1.dat.data_ro[:])
