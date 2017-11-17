"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

from __future__ import print_function

from fenics import *
from fenics_adjoint import *

from numpy.random import rand

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic.copy(deepcopy=True)
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    forward = main(ic, annotate=True)

    if False:
        # TODO: Not implemented.
        replay_dolfin(forget=False)

    J = assemble(forward*forward*dx + ic*ic*dx)
    dJdic = compute_gradient(J, Control(ic))

    def Jfunc(ic):
        forward = main(ic, annotate=False)
        return assemble(forward*forward*dx + ic*ic*dx)

    HJic = Hessian(J, Control(ic))
    h = Function(V)
    h.vector()[:] = rand(V.dim())

    dJdic = h._ad_dot(dJdic)
    HJic = h._ad_dot(HJic(h))

    minconv = taylor_test(Jfunc, ic, h, dJdic, Hm=HJic)
    assert minconv > 2.7
