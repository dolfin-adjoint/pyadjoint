from __future__ import print_function
import random

from dolfin import *
from dolfin_adjoint import *
import sys
import libadjoint.exceptions

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

def main(ic, annotate=False):
    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, "-1.0", "on_boundary")

    mass = inner(u, v)*dx
    soln = Function(V)

    solve(mass == action(mass, ic), soln, bc, annotate=annotate)
    return soln

def propagator(ic): # a hand-coded tangent linear model, essentially
    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, "0.0", "on_boundary")

    mass = inner(u, v)*dx
    soln = Function(V)

    solve(mass == action(mass, ic), soln, bc, annotate=False)
    return soln

if __name__ == "__main__":

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=4), V)
    soln = main(ic, annotate=True)

    try:
        svd = compute_gst(ic, soln, 1, ic_norm=None, final_norm=None)
    except libadjoint.exceptions.LibadjointErrorSlepcError:
        info_red("Not testing since SLEPc unavailable.")
        import sys; sys.exit(0)

    assert svd.ncv >= 1
    (sigma, u, v, error) = svd.get_gst(0, return_vectors=True, return_residual=True)

    print("Maximal singular value: ", sigma)

    Lv = propagator(v)
    residual = (Lv.vector() - sigma*u.vector()).norm("l2")
    print("Residual: ", residual)

    assert residual < 1.0e-14
