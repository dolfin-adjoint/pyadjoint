"""
Naive implementation of Burgers' equation, goes oscillatory later
"""
from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def Dt(u, u_, timestep):
    return (u - u_) / timestep


def test_burgers_picard():
    n = 2
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 1)

    ic = project(Constant(1.0), V)
    u_ = Function(V, ic, name="Solution")
    m = Control(u_)
    u = TrialFunction(V)
    v = TestFunction(V)

    nu = Constant(0.0001)
    timestep = Constant(1.0)

    F = (Dt(u, u_, timestep)*v
         + u_*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    (a, L) = system(F)

    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 3.0 # set to 1.0 - eps to compare against manual_hessian.py
    u = Function(V)

    while (t <= end):
        solve(a == L, u, bc)

        u_.assign(u)
        t += float(timestep)

    forward = u_

    J = assemble(inner(forward, forward)**2*dx)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJdm = compute_gradient(J, m)
    dJdm = h._ad_dot(dJdm)
    HJm  = Hessian(J, m)
    Hm = h._ad_dot(HJm(h))

    assert taylor_test(ReducedFunctional(J, m), ic, h, dJdm=dJdm, Hm=Hm) > 1.9