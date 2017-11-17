"""Verify the tangent linear model of example 5.2 in 10.1007/s10589-009-9307-9"""

from fenics import *
from fenics_adjoint import *

parameters["form_compiler"]["representation"] = "uflacs"

n = 32
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
gamma = Constant(1e3) # 1.0 / alpha in the paper

def main():
    y = Function(V, name="Solution")
    u = Function(V, name="Control")
    w = TestFunction(V)

    eps = 1e-4

    def smoothmax(r):
        return conditional(gt(r, eps), r - eps/2, conditional(lt(r, 0), 0, r**2 / (2*eps)))
    def uflmax(a, b):
        return conditional(gt(a, b), a, b)

    f = interpolate(Expression("-std::abs(x[0]*x[1] - 0.5) + 0.25", degree=1), V)
    F = inner(grad(y), grad(w))*dx - gamma * inner(smoothmax(-y), w)*dx - inner(f + u, w)*dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    solve(F == 0, y, bcs = bc, solver_parameters={"newton_solver": {"maximum_iterations": 30}})
    return y

if __name__ == "__main__":
    def J(g):
        gamma.assign(g)
        y = main()
        return assemble(inner(y, y)*dx)

    y = main()
    m = Control(gamma)

    Jy = inner(y, y)*dx
    dJdu = assemble(derivative(Jy, y))
    Jm = assemble(Jy)
    for (soln, var) in compute_tlm(m):
        pass
    dJdm = soln.vector().inner(dJdu)

    minconv = taylor_test(J, m, Jm, dJdm)
    assert minconv > 1.89
