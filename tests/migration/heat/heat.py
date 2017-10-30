from fenics import *
from fenics_adjoint import *

from numpy.random import rand


f = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=4)
mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)

def main(ic, annotate=True):
    u = TrialFunction(V)
    v = TestFunction(V)

    u_0 = Function(V, name="Solution")
    u_0.assign(ic, annotate=annotate)

    u_1 = Function(V, name="NextSolution")

    dt = Constant(0.1)

    F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx

    bc = DirichletBC(V, 1.0, "on_boundary")

    a, L = lhs(F), rhs(F)

    t = float(dt)
    T = 1.0
    n = 1

    while t <= T:

        solve(a == L, u_0, bc, annotate=annotate, solver_parameters={"linear_solver": "cg", "preconditioner": "ilu", "krylov_solver": {"absolute_tolerance": 1.0e-16, "relative_tolerance": 1.0e-200}})
        t += float(dt)

    return u_0

if __name__ == "__main__":

    ic = Function(V, name="InitialCondition")
    u = main(ic)

    J = assemble(u*u*u*u*dx)
    m = Control(ic)
    dJdm = compute_gradient(J, m)
    HJm  = Hessian(J, m)
    h = Function(V)
    h.vector()[:] = rand(V.dim())*5000

    dJdm = h._ad_dot(dJdm)
    HJm = h._ad_dot(HJm(h))

    def J(ic):
        u = main(ic, annotate=False)
        return assemble(u*u*u*u*dx)

    minconv = taylor_test(J, ic, h, dJdm, Hm=HJm)
    assert minconv > 2.9
