from __future__ import print_function

from fenics import *
from fenics_adjoint import *

from numpy.random import rand

n = 3
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 1)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic.copy(deepcopy=True)
    u = TrialFunction(V)
    v = TestFunction(V)

    mass = assemble(inner(u, v) * dx)
    if annotate: assert hasattr(mass, 'form')

    advec = assemble(u_*u.dx(0)*v * dx)
    if annotate: assert hasattr(advec, 'form')

    rhs = assemble(inner(u_, v) * dx)
    if annotate: assert hasattr(rhs, 'form')

    L = mass + advec

    if annotate: assert hasattr(L, 'form')
    solve(L, u_.vector(), rhs, 'lu', annotate=annotate)

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    forward = main(ic, annotate=True)

    m = Control(ic)
    J = assemble(forward*forward*dx)
    dJdm = compute_gradient(J, m)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJdm = h._ad_dot(dJdm)

    def Jfunc(ic):
        forward = main(ic, annotate=False)
        return assemble(forward*forward*dx)

    minconv = taylor_test(Jfunc, ic, h, dJdm)
    assert minconv > 1.8
