"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake_adjoint import *

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

    F = (inner(Dt(u, ic, timestep), v)
         + u*inner(u.dx(0), v) + nu*inner(u.dx(0), v.dx(0)))*dx
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

    F = (inner(Dt(u, u_, timestep), v)
         + u*inner(u.dx(0), v) + nu*inner(u.dx(0), v.dx(0)))*dx

    end = 0.2
    while (t <= end):
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)

        t += float(timestep)

    return assemble(u_**2*dx + ic**2*dx)


@pytest.mark.parametrize("solve_type",
                         ["solve", "NLVS"])
def test_burgers_newton(solve_type):
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2*pi*x),  V)

    val = J(ic, solve_type)

    Jhat = ReducedFunctional(val, Control(ic))

    h = Function(V)
    h.assign(1, annotate=False)
    assert taylor_test(Jhat, ic, h) > 1.9
