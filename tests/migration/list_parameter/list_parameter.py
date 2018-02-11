"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

import sys

from fenics import *
from fenics_adjoint import *

from numpy.random import rand

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, nu, annotate=False):

    u_ = ic.copy(deepcopy=True)
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

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
    nu = Constant(0.0001)
    forward = main(ic, nu, annotate=True)

    J = assemble(forward*forward*dx + ic*ic*dx)
    m = [Control(ic), Control(nu)]
    dJdm = compute_gradient(J, m)
    h1 = Function(V)
    h1.vector()[:] = rand(V.dim())
    h2 = Constant(0.0001)

    hs = [h1, h2]
    dJdm = sum([hs[i]._ad_dot(dJdm[i]) for i in range(len(hs))])

    def Jfunc(m):
        lic, lnu = m
        forward = main(lic, lnu, annotate=False)
        return assemble(forward*forward*dx + lic*lic*dx)

    minconv = taylor_test(Jfunc, [ic, nu], hs, dJdm=dJdm)
    assert minconv > 1.7
