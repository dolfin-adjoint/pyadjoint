"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

import sys

from dolfin import *
from dolfin_adjoint import *

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, nu, annotate=False):

    u_ = ic.copy(deepcopy=True, name="Velocity")
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
        adj_inc_timestep()

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    nu = Constant(0.0001, name="nu")

    forward = main(ic, nu, annotate=True)

    J = Functional(forward*forward*dx*dt[FINISH_TIME] + forward*forward*dx*dt[START_TIME])
    Jm = assemble(forward*forward*dx + ic*ic*dx)
    m = [FunctionControl("Velocity"), Control(nu)]
    dJdm = compute_gradient_tlm(J, m, forget=False)

    def Jfunc(m):
        lic, lnu = m

        forward = main(lic, lnu, annotate=False)
        return assemble(forward*forward*dx + lic*lic*dx)

    minconv = taylor_test(Jfunc, m, Jm, dJdm)
    assert minconv > 1.7
