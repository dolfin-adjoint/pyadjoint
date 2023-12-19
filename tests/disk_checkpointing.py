from firedrake import *
from pyadjoint import (ReducedFunctional, get_working_tape, stop_annotating,
                       pause_annotation, Control)
from checkpoint_schedules import (
    Revolve, MultistageCheckpointSchedule, SingleMemoryStorageSchedule,
    NoneCheckpointSchedule, MixedCheckpointSchedule, StorageType, HRevolve)
import numpy as np

from firedrake.adjoint import *

def Dt(u, u_, timestep):
    return (u - u_) / timestep

def J(ic, solve_type):
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
        step = int(t/float(timestep))
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)

    return assemble(u_ * u_ * dx + ic * ic * dx)

continue_annotation()
enable_disk_checkpointing()
n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)
end = 0.3
timestep = Constant(1.0 / n)
steps = int(end / float(timestep))
tape = get_working_tape()
tape.progress_bar = ProgressBar
# tape.enable_checkpointing(HRevolve(steps, 1, 10))
x, = SpatialCoordinate(mesh)
ic = project(sin(2. * pi * x), V)

val = J(ic, "NLVS")
pause_disk_checkpointing()
J_hat = ReducedFunctional(val, Control(ic))
J_hat(ic)