from dolfin import *
from dolfin_adjoint import *

dolfin.parameters["adjoint"]["record_all"] = True

import random
import sys

mesh = UnitIntervalMesh(5)
V2 = FunctionSpace(mesh, "CG", 2)
V1 = FunctionSpace(mesh, "CG", 1)

def main(ic, annotate=False):

    ic_proj = Function(V2, name="Projection")
    u = TrialFunction(V2)
    v = TestFunction(V2)
    solve(inner(u, v)*dx == inner(ic, v)*dx, ic_proj, annotate=annotate)

    out = interpolate(ic_proj, V1, annotate=annotate)
    return out

if __name__ == "__main__":
    ic = Function(V2, name="InitialCondition")
    vec = ic.vector()
    for i in range(len(vec)):
        vec[i] = random.random()

    out = main(ic, annotate=True)

    success = replay_dolfin()

#  J = Functional(out*out*dx*dt[FINISH_TIME])
#  icparam = FunctionControl("InitialCondition")
#  dJdic = compute_gradient(J, icparam)
#
#  def J(ic):
#    out = main(ic, annotate=False)
#    return assemble(out*out*dx)
#
#  minconv = utils.test_initial_condition_adjoint(J, ic, dJdic)
#  if minconv < 1.9:
#    sys.exit(1)

    if not success:
        sys.exit(1)
