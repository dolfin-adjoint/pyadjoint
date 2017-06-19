"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""
from firedrake import *
from firedrake_adjoint import *

parameters["pyop2_options"]["lazy_evaluation"] = False

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


def test_burgers_newton():
    x, = SpatialCoordinate(mesh)
    ic = project(sin(2*pi*x),  V)

    val = J(ic)

    Jhat = ReducedFunctional(val, ic)

    h = Function(V)
    h.assign(1, annotate=False)
    ic = Function(ic)
    assert taylor_test(Jhat, ic, h) > 1.9
