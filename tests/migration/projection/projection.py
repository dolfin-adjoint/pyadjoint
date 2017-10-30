from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitSquareMesh(4, 4)
V3 = FunctionSpace(mesh, "CG", 3)
V2 = FunctionSpace(mesh, "CG", 2)


def main(ic, annotate=False):
    bc = DirichletBC(V2, "-1.0", "on_boundary")
    soln = project(ic, V2, bcs=bc, solver_type='lu', annotate=annotate)
    return soln

if __name__ == "__main__":

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=4), V3)
    soln = main(ic, annotate=True)


    if False:
        # TODO: Not implemented.
        replay_dolfin()

    J = assemble(soln*soln*dx)
    dJdic = compute_gradient(J, Control(ic))
    h = Function(V3)
    h.vector()[:] = rand(V3.dim())
    dJdic = h._ad_dot(dJdic)

    def J(ic):
        soln = main(ic, annotate=False)
        return assemble(soln*soln*dx)

    minconv = taylor_test(J, ic, h, dJdic)
    assert minconv > 1.9
