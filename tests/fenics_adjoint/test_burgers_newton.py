"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

set_log_level(LogLevel.CRITICAL)

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
        problem = NonlinearVariationalProblem(F, u, bcs=bc, J=derivative(F, u))
        solver = NonlinearVariationalSolver(problem)
        solver.solve()
    else:
        solve(F == 0, u, bc)
    u_.assign(u)
    t += float(timestep)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    end = 0.2
    while (t <= end):
        if solve_type == "NLVS":
            solver.solve()
        else:
            solve(F == 0, u, bc)
        u_.assign(u)

        t += float(timestep)

    return assemble(u_*u_*dx + ic*ic*dx)

def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))

    return r

@pytest.mark.parametrize("solve_type",
                         ["solve", "NLVS"])
def test_burgers_newton(solve_type):
    pr = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    ic = Function(V)
    ic.vector()[:] = pr.vector()[:]
    c = Control(ic)
    _test_adjoint(J, ic, c, solve_type)

def _test_adjoint(J, f, control, solve_type):
    import numpy.random
    tape = Tape()
    set_working_tape(tape)

    V = f.function_space()
    h = Function(V)
    h.vector()[:] = numpy.random.rand(V.dim())
    pertubed_ic = Function(V)

    eps_ = [0.01/(2.0**i) for i in range(5)]
    residuals = []
    for eps in eps_:

        pertubed_ic.vector()[:] = f.vector()[:]
        pertubed_ic.vector()[:] += eps*h.vector()[:]
        Jp = J(pertubed_ic, solve_type)
        tape.clear_tape()
        Jm = J(f, solve_type)
        Jm.block_variable.adj_value = 1.0
        tape.evaluate_adj()

        dJdf = control.adj_value

        residual = abs(Jp - Jm - eps*dJdf.inner(h.vector()))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    print(r)

    tol = 1E-1
    assert( r[-1] > 2-tol )
