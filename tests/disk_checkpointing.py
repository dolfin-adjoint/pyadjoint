from firedrake import *
from pyadjoint import (ReducedFunctional, get_working_tape, stop_annotating,
                       pause_annotation, Control)
from checkpoint_schedules import (
    Revolve, MultistageCheckpointSchedule, SingleMemoryStorageSchedule,
    NoneCheckpointSchedule, MixedCheckpointSchedule, StorageType)
import numpy as np

from firedrake.adjoint import *

def Dt(u, u_, timestep):
    return (u - u_) / timestep

def test_disk_checkpointing():
    tape = get_working_tape()
    tape.clear_tape()
    enable_disk_checkpointing()

    mesh = checkpointable_mesh(UnitSquareMesh(10, 10, name="mesh"))
    J_disk, grad_J_disk = adjoint_example(mesh)
    tape.clear_tape()
    pause_disk_checkpointing()

    J_mem, grad_J_mem = adjoint_example(mesh)

    assert np.allclose(J_disk, J_mem)
    assert np.allclose(assemble((grad_J_disk - grad_J_mem)**2*dx), 0.0)
    tape.clear_tape()



def J(ic, solve_type, checkpointing):
    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)
    u_.assign(ic)
    nu = Constant(0.0001)
    F = (Dt(u, u_, timestep) * v
         + u * u.dx(0) * v + nu * u.dx(0) * v.dx(0)) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    if solve_type == "NLVS":
        problem = NonlinearVariationalProblem(F, u, bcs=bc)
        solver = NonlinearVariationalSolver(problem)

    tape = get_working_tape()
    t = 0
    for t in tape.timestepper(np.arange(t, end, float(timestep))):
        storage_type = tape.timesteps[int(t)].checkpoint_storage_type
        if storage_type == StorageType.DISK:
            enable_disk_checkpointing()

        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)
        if storage_type == StorageType.DISK:
            pause_disk_checkpointing()

    return assemble(u_ * u_ * dx + ic * ic * dx)

continue_annotation()
n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)
end = 0.3
timestep = Constant(1.0 / n)
steps = int(end / float(timestep))
tape = get_working_tape()
tape.progress_bar = ProgressBar
tape.enable_checkpointing(Revolve(steps, steps // 3))
x, = SpatialCoordinate(mesh)
ic = project(sin(2. * pi * x), V)

J(ic, "NLVS", "Revolve")