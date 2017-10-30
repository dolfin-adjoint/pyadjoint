from __future__ import print_function
import random

from dolfin import *
from dolfin_adjoint import *
import sys
import libadjoint.exceptions

mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, "CG", 1)
dolfin.parameters["adjoint"]["record_all"] = True

def main(ic, annotate=False):
    u = TrialFunction(V)
    v = TestFunction(V)

    mass = inner(u, v)*dx
    soln = Function(V)

    solve(mass == action(mass, ic), soln, annotate=annotate,
            solver_parameters={"linear_solver": "cg"})
    return soln

if __name__ == "__main__":

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=1), V)
    soln = main(ic, annotate=True)

    parameters["adjoint"]["cache_factorizations"] = True

    try:
        svd = compute_gst(ic, soln, 5, ic_norm="mass", final_norm="mass")
    except libadjoint.exceptions.LibadjointErrorSlepcError:
        info_red("Not testing since SLEPc unavailable.")
        import sys; sys.exit(0)

    (sigma, u, v, error) = svd.get_gst(0, return_vectors=True, return_residual=True)

    print("Maximal singular value: ", (sigma, error))

    u_l2 = assemble(inner(u, u)*dx)
    print("L2 norm of u: %.16e" % u_l2)
    assert abs(u_l2 - 1.0) < 1.0e-13

    v_l2 = assemble(inner(v, v)*dx)
    print("L2 norm of v: %.16e" % v_l2)
    assert abs(v_l2 - 1.0) < 1.0e-13

    assert abs(sigma - 1.0) < 1.0e-13
