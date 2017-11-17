from fenics import *
from fenics_adjoint import *

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 3)
a = Constant(2.0)
b = Constant(3.0)

def main(ic, params, annotate=False):
    u = TrialFunction(V)
    v = TestFunction(V)
    (a, b) = params

    bc = DirichletBC(V, "-1.0", "on_boundary")

    mass = inner(u, v)*dx
    rhs = a*b*action(mass, ic)
    soln = Function(V)
    da = Function(V)

    solve(mass == rhs, soln, bc, annotate=annotate)
    return soln

if __name__ == "__main__":

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=1), V)
    soln = main(ic, (a, b), annotate=True)

    J = assemble(soln*soln*dx)
    dJda = compute_gradient(J, [Control(a), Control(b)])
    hs = [Constant(1.0), Constant(1.0)]
    dJda = sum([hs[i]._ad_dot(dJda[i]) for i in range(len(hs))])

    def J(params):
        soln = main(ic, params, annotate=False)
        return assemble(soln*soln*dx)

    minconv = taylor_test_multiple(J, [a, b], hs, dJda)
    assert minconv > 1.9
