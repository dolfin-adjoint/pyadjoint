"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

set_log_level(CRITICAL)

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def J(ic):
    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, ic, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    solve(F == 0, u, bc)
    u_.assign(u)
    t += float(timestep)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    end = 0.2
    while (t <= end):
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

def test_burgers_newton():
    pr = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    ic = Function(V)
    ic.vector()[:] = pr.vector()[:]

    _test_adjoint(J, ic)

def _test_adjoint(J, f):
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
        Jp = J(pertubed_ic)
        tape.clear_tape()
        Jm = J(f)
        Jm.set_initial_adj_input(1.0)
        tape.evaluate()

        dJdf = f.get_adj_output()
        #print dJdf.array()

        residual = abs(Jp - Jm - eps*dJdf.inner(h.vector()))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    print(r)
    #print residuals

    tol = 1E-1
    assert( r[-1] > 2-tol )
