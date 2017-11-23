"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

from __future__ import print_function
import sys
import numpy
import random

from dolfin import *
from dolfin_adjoint import *
import libadjoint.exceptions

dolfin.parameters["adjoint"]["record_all"] = True
dolfin.parameters["adjoint"]["fussy_replay"] = False

n = 50
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 1)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic.copy(deepcopy=True, name="State")
    u = Function(V, name="NextState")
    v = TestFunction(V)

    nu = Constant(0.1)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
        + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 1.0
    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep()

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    ic_copy = ic.copy(deepcopy=True)
    forward = main(ic, annotate=True)
    parameters["adjoint"]["stop_annotating"] = True
    forward_copy = forward.copy(deepcopy=True)

    ic = forward
    ic.vector()[:] = ic_copy.vector()
    factor = 0.01

    try:
        svd = compute_gst("State", "State", nsv=1, ic_norm=None, final_norm=None)
    except libadjoint.exceptions.LibadjointErrorSlepcError:
        info_red("Not testing since SLEPc unavailable.")
        import sys; sys.exit(0)

    (sigma, u, v) = svd.get_gst(0, return_vectors=True)

    ic_norm = v.vector().norm("l2")

    perturbed_ic = ic.copy(deepcopy=True)
    perturbed_ic.vector().axpy(factor, v.vector())
    perturbed_soln = main(perturbed_ic, annotate=False)

    final_norm = (perturbed_soln.vector() - forward_copy.vector()).norm("l2")/factor
    print("Norm of initial perturbation: ", ic_norm)
    print("Norm of final perturbation: ", final_norm)
    ratio = final_norm / ic_norm
    print("Ratio: ", ratio)
    print("Predicted growth of perturbation: ", sigma)

    prediction_error = abs(sigma - ratio)/ratio * 100
    print("Prediction error: ", prediction_error,  "%")
    assert prediction_error < 2

    try:
        svd = compute_gst("State", "State", nsv=1, ic_norm="mass", final_norm="mass")
    except libadjoint.exceptions.LibadjointErrorSlepcError:
        info_red("Not testing since SLEPc unavailable.")
        import sys; sys.exit(0)

    (sigma, u, v) = svd.get_gst(0, return_vectors=True)

    ic_norm = sqrt(assemble(inner(v, v)*dx))

    perturbed_ic = ic.copy(deepcopy=True)
    perturbed_ic.vector().axpy(factor, v.vector())
    perturbed_soln = main(perturbed_ic, annotate=False)

    final_norm = sqrt(assemble(inner(perturbed_soln - forward_copy, perturbed_soln - forward_copy)*dx))/factor
    print("Norm of initial perturbation: ", ic_norm)
    print("Norm of final perturbation (after solve): ", final_norm)
    print("Norm of final perturbation (from SVD): ", sqrt(assemble(inner(u, u)*dx)))
    ratio = final_norm / ic_norm
    print("Ratio: ", ratio)
    print("Predicted growth of perturbation: ", sigma)

    prediction_error = abs(sigma - ratio)/ratio * 100
    print("Prediction error: ", prediction_error,  "%")
    assert prediction_error < 2
