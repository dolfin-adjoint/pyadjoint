"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
# import pytest
# pytest.importorskip("firedrake")
from checkpoint_schedules import *
import numpy as np
import firedrake as fd
import firedrake.adjoint as fda

fd.set_log_level(fd.CRITICAL)
fda.continue_annotation()

tape = fda.get_working_tape()
tape.progress_bar = fd.ProgressBar


n = 10
mesh = fd.UnitIntervalMesh(n)
V = fd.FunctionSpace(mesh, "CG", 2)


def Dt(u, u_, timestep):
    return (u - u_)/timestep


def J(ic, solve_type):
    u_ = fd.Function(V)
    u = fd.Function(V)
    v = fd.TestFunction(V)

    nu = fd.Constant(0.0001)

    timestep = fd.Constant(1.0/n)

    F = (Dt(u, ic, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*fd.dx
    bc = fd.DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    if solve_type == "NLVS":
        problem = fd.NonlinearVariationalProblem(F, u, bcs=bc)
        solver = fd.NonlinearVariationalSolver(problem)
        solver.solve()
    else:
        fd.solve(F == 0, u, bc)
    u_.assign(u)
    tape = fda.get_working_tape()
    tape.end_timestep()
    t += float(timestep)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*fd.dx

    end = 10 * 0.1
    for t in tape.timestepper(np.arange(t, end, float(timestep))):
        if solve_type == "NLVS":
            solver.solve()
        else:
            fd.solve(F == 0, u, bc)
        u_.assign(u)
        if t == end:
            break
    return fd.assemble(u_*u_*fd.dx + ic*ic*fd.dx)


# @pytest.mark.parametrize("solve_type",
#                          ["solve", "NLVS"])
def test_burgers_newton(solve_type):
    tape.enable_checkpointing(Revolve(10, 1), n_steps=10)
    x, = fd.SpatialCoordinate(mesh)
    ic = fd.project(fd.sin(2.*fd.pi*x), V)

    val = J(ic, solve_type)
    # dJdc = fda.compute_gradient(val, fda.Control(ic))
    # assert len(tape.timesteps) == 3
    Jhat = fda.ReducedFunctional(val, fda.Control(ic))
    dJ = Jhat.derivative()
    assert np.allclose(Jhat(ic), val)

    dJbar = Jhat.derivative()
    assert np.allclose(dJ.dat.data_ro[:], dJbar.dat.data_ro[:])

    h = fd.Function(V)
    h.assign(1, annotate=False)
    assert fda.taylor_test(Jhat, ic, h) > 1.9


test_burgers_newton("solve")
