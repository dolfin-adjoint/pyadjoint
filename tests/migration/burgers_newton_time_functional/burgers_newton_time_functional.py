"""
Implementation of Burger's equation with nonlinear solve in each
timestep and a functional integrating over time
"""

from __future__ import print_function

from fenics import *
from fenics_adjoint import *

from numpy.random import rand, seed
seed(21)

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
    j = 0
    j += 0.5*AdjFloat(timestep)*assemble(u_*u_*u_*u_*dx)

    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

        if t>end:
            quad_weight = 0.5
        else:
            quad_weight = 1.0
        j += quad_weight*AdjFloat(timestep)*assemble(u_*u_*u_*u_*dx)

    return j, u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    ic_copy = ic.copy(deepcopy=True, annotate=False)
    j, forward = main(ic, annotate=True)
    forward_copy = forward.copy(deepcopy=True, annotate=False)

    J = j
    m = Control(ic)
    dJdm = compute_gradient(J, m)

    h = Function(V)
    h.vector()[:] = rand(V.dim())*1.4
    dJdm = h._ad_dot(dJdm)
    HJm = compute_hessian(J, m, h)
    HJm = h._ad_dot(HJm)

    def Jfunc(ic):
        j, forward = main(ic, annotate=False)
        return j

    minconv = taylor_test(Jfunc, ic_copy, h, dJdm, Hm=HJm)
    assert minconv > 2.7
