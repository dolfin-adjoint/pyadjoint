"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

from fenics import *
from fenics_adjoint import *

from numpy.random import rand

n = 2
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 1)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = Function(V, name="Solution")
    u_.assign(ic)
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
        solve(a == L, u, bc, annotate=annotate)

        u_.assign(u, annotate=annotate)
        t += float(timestep)

    return u_

if __name__ == "__main__":

    ic = project(Constant(1.0),  V)
    forward = main(ic, annotate=True)

    J = assemble(inner(forward, forward)**2*dx)
    m = Control(ic)

    dJdm = compute_gradient(J, m)
    HJm  = Hessian(J, m)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJdm = h._ad_dot(dJdm)
    HJm = h._ad_dot(HJm(h))

    def Jfunc(ic):
        forward = main(ic, annotate=False)
        return assemble(inner(forward, forward)**2*dx)

    minconv = taylor_test(Jfunc, ic, h, dJdm, Hm=HJm)
    assert minconv > 2.9
