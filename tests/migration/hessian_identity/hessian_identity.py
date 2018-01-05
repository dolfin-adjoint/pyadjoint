from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "R", 0)

test = TestFunction(V)
trial = TrialFunction(V)

def main(m):
    u = interpolate(Constant(0.1), V)
    u = Function(V, u.vector())

    F = inner(u*u, test)*dx - inner(m, test)*dx
    solve(F == 0, u)
    F = inner(sin(u)*u*u*trial, test)*dx - inner(u**4, test)*dx
    solve(lhs(F) == rhs(F), u)

    return u

if __name__ == "__main__":
    m = interpolate(Constant(2.13), V)
    m = Function(V, m.vector())
    u = main(m)

    J = assemble(inner(u, u)**3*dx + inner(m, m)*dx)
    dJdm = compute_gradient(J, Control(m))
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJdm = h._ad_dot(dJdm)
    HJm = compute_hessian(J, Control(m), h)
    HJm = h._ad_dot(HJm)

    def Jhat(m):
        u = main(m)
        return assemble(inner(u, u)**3*dx + inner(m, m)*dx)

    minconv = taylor_test(Jhat, m, h, dJdm, Hm=HJm)
    assert minconv > 2.9
