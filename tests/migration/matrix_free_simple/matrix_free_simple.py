from dolfin import *
from dolfin_adjoint import *

import sys

dolfin.parameters["adjoint"]["record_all"] = True

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, 'CG', 1)
bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
u = TrialFunction(V); v = TestFunction(V);

def main(ic):
    b = assemble(ic * v* dx)
    bc.apply(b)

    x = Function(V)
    KrylovSolver = AdjointPETScKrylovSolver("cg","none")
    mat = AdjointKrylovMatrix(inner(grad(u), grad(v))*dx, bcs=bc)
    KrylovSolver.solve(mat, down_cast(x.vector()), down_cast(b))

    return x

if __name__ == "__main__":

    # There must be a better way of doing this ...
    import random
    ic = Function(V)
    icvec = ic.vector()
    for i in range(len(icvec)):
        icvec[i] = random.random()

    iccopy = Function(ic)
    final = main(ic)

    adj_html("forward.html", "forward")
    replay_dolfin()

    J = Functional(inner(final, final)*dx*dt[FINISH_TIME])
    for (adjoint, var) in compute_adjoint(J, forget=False):
        pass

    def J(ic):
        soln = main(ic)
        return assemble(inner(soln, soln)*dx)

    minconv = utils.test_initial_condition_adjoint(J, ic, adjoint, seed=1.0e-3)
    if minconv < 1.9:
        sys.exit(1)
