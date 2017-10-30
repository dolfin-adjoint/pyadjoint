from dolfin import *
from dolfin_adjoint import *

dolfin.parameters["adjoint"]["record_all"] = True

import random
import sys

if dolfin.__version__[0:3] > '1.1':
    sys.exit(0)

mesh = UnitIntervalMesh(40)
V = FunctionSpace(mesh, "CG", 2)

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < DOLFIN_EPS and on_boundary

    def map(self, x, y):
        y[0] = x[0] - 1

def main(ic, annotate=False):

    ic_proj = Function(V, name="Projection")
    out = Function(V, name="ProjectionAssigned")
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = PeriodicBC(V, PeriodicBoundary())
    solve(inner(u, v)*dx == inner(ic, v)*dx, ic_proj, bc, annotate=annotate)

    F = inner(out, v)*dx - inner(ic_proj, v)*dx
    solve(F == 0, out, bc, annotate=annotate)

    return out

if __name__ == "__main__":
    ic = Function(V, name="InitialCondition")
    vec = ic.vector()
    for i in range(len(vec)):
        vec[i] = random.random()

    out = main(ic, annotate=True)
    success = replay_dolfin(tol=1.0e-15)

    if not success:
        sys.exit(1)

    J = Functional(out*out*dx*dt[FINISH_TIME] + ic*ic*dx*dt[START_TIME])
    icparam = FunctionControl("InitialCondition")
    dJdic = compute_gradient(J, icparam)

    def J(ic):
        out = main(ic, annotate=False)
        return assemble(out*out*dx + ic*ic*dx)

    minconv = utils.test_initial_condition_adjoint(J, ic, dJdic)
    if minconv < 1.9:
        sys.exit(1)
