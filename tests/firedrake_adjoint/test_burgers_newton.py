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

def basics():
    n = 30
    mesh = UnitIntervalMesh(n)
    end = 0.3
    timestep = Constant(1.0/n)
    steps = int(end/float(timestep)) + 1
    return mesh, timestep, steps

def Dt(u, u_, timestep):
    return (u - u_)/timestep


def J(ic, solve_type, timestep, steps, V):
    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)
    u_.assign(ic)
    nu = Constant(0.0001)
    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    if solve_type == "NLVS":
        problem = NonlinearVariationalProblem(F, u, bcs=bc)
        solver = NonlinearVariationalSolver(problem)

    tape = get_working_tape()
    for _ in tape.timestepper(range(steps)):
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)

    return assemble(u_*u_*dx + ic*ic*dx)


@pytest.mark.parametrize("solve_type, checkpointing",
                         [
                          ("solve", "Revolve"),
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
    mesh, timestep, steps = basics()
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    if checkpointing:
        if checkpointing == "Revolve":
            schedule = Revolve(steps, steps//3)
        if checkpointing == "SingleMemory":
            schedule = SingleMemoryStorageSchedule()
        if checkpointing == "Mixed":
            schedule = MixedCheckpointSchedule(steps, steps//3, storage=StorageType.DISK)
        if checkpointing == "NoneAdjoint":
            schedule = NoneCheckpointSchedule()
        if schedule.uses_storage_type(StorageType.DISK):
            manage_disk_checkpointing = AdjointDiskCheckpointing()
        else:
            manage_disk_checkpointing = None
        tape.enable_checkpointing(
            schedule, manage_disk_checkpointing=manage_disk_checkpointing)
        if schedule.uses_storage_type(StorageType.DISK):
            mesh = checkpointable_mesh(mesh)
    x, = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)
    ic = project(sin(2. * pi * x), V)
    val = J(ic, solve_type, timestep, steps, V)
    if checkpointing:
        assert len(tape.timesteps) == steps
    Jhat = ReducedFunctional(val, Control(ic))
    if checkpointing != "NoneAdjoint":
        dJ = Jhat.derivative()

    # Recomputing the functional with a modified control variable
    # before the recompute test.
    Jhat(project(sin(pi*x), V))

    # Recompute test
    assert(np.allclose(Jhat(ic), val))
    if checkpointing != "NoneAdjoint":
        dJbar = Jhat.derivative()
        # Test recompute adjoint-based gradient
        assert np.allclose(dJ.dat.data_ro[:], dJbar.dat.data_ro[:])
        # Taylor test
        assert taylor_test(Jhat, ic, Function(V).assign(1, annotate=False)) > 1.9


@pytest.mark.parametrize("solve_type, checkpointing",
                         [("solve", "Revolve"),
                          ("NLVS", "Revolve"),
                          ("solve", "Mixed"),
                          ("NLVS", "Mixed"),
                          ])
def test_checkpointing_validity(solve_type, checkpointing):
    """Compare forward and backward results with and without checkpointing.
    """
    mesh, timestep, steps = basics()
    V = FunctionSpace(mesh, "CG", 2)
    # Without checkpointing
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2.*pi*x), V)

    val0 = J(ic, solve_type, timestep, steps, V)
    Jhat = ReducedFunctional(val0, Control(ic))
    dJ0 = Jhat.derivative()
    tape.clear_tape()

    # With checkpointing
    tape.progress_bar = ProgressBar
    if checkpointing == "Revolve":
        tape.enable_checkpointing(Revolve(steps, steps//3))
    if checkpointing == "Mixed":
        tape.enable_checkpointing(MixedCheckpointSchedule(steps, steps//3, storage=StorageType.DISK),
                                  manage_disk_checkpointing=AdjointDiskCheckpointing())
        mesh = checkpointable_mesh(mesh)
    V = FunctionSpace(mesh, "CG", 2)
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2.*pi*x), V)
    val1 = J(ic, solve_type, timestep, steps, V)
    Jhat = ReducedFunctional(val1, Control(ic))
    assert len(tape.timesteps) == steps
    assert np.allclose(val0, val1)
    assert np.allclose(dJ0.dat.data_ro[:], Jhat.derivative().dat.data_ro[:])
