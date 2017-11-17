import sys

from fenics import *
from fenics_adjoint import *

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)
f = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=4), V)

def main(f, annotate=True):
    u = TrialFunction(V)
    v = TestFunction(V)

    u = Function(V, name="Solution")
    F = (inner(grad(u), grad(v)) + f*v)*dx
    bc = DirichletBC(V, 1.0, "on_boundary")
    solve(F == 0, u, bc, annotate=annotate)
    return u

if __name__ == "__main__":

    u = main(f)
    m = Control(f)

    J1 = assemble(u**2 * dx)
    dJ1dm = compute_gradient(J1, m)
    HJ1m  = Hessian(J1, m)


    J2 = assemble(f**2 * dx)
    dJ2dm = compute_gradient(J2, m)
    HJ2m  = Hessian(J2, m)

    J = 0.5 * J1 + (-1) * (-J2) / 2
    dJdm = compute_gradient(J, m)
    HJm  = Hessian(J, m)

    assert (dJdm.vector() - 0.5 * dJ1dm.vector() - 0.5 * dJ2dm.vector()).norm("l2") < 1e-14
    assert (HJm(f).vector() - 0.5 * HJ1m(f).vector() - 0.5 * HJ2m(f).vector()).norm("l2") < 1e-14
