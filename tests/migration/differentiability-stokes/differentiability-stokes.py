"""This script implements the Taylor remainder convergence test
for an individual form.

Suppose we have an expression F(T) that is a function of T. We
can check the correctness of the derivative dF/dT by noting that

||F(T + dT) - F(T)|| converges at first order

but that

||F(T + dT) - F(T) - dF/dT . dT|| converges at second order.

In this example, F(T) is the action of the Stokes operator
on a supplied velocity field:

F(T) = action(momentum(T), u).

"""

from __future__ import print_function
from numpy import random
from dolfin import *
from math import log
from logging import INFO
import sys

# Uncomment the line below to make this script work.
# Any positive value works (but -1 does not).
#parameters["form_compiler"]["quadrature_degree"] = 5
parameters["form_compiler"]["log_level"] = INFO
parameters["form_compiler"]["representation"] = "quadrature"

mesh = Mesh("mesh.xml.gz")
cg2 = VectorElement("CG", triangle, 2)
cg1 = FiniteElement("CG", triangle, 1)
cg2cg1 = MixedElement([cg2, cg1])

V = FunctionSpace(mesh, cg1)
W = FunctionSpace(mesh, cg2cg1)

T = Function(V, "temperature.xml.gz")
w = Function(W, "velocity.xml.gz")

def form(T):
    eta = exp(-log(1000)*T)
    Ra = 10000
    H = Ra*T
    g = Constant((0.0, -1.0))

    # Define basis functions
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    strain = lambda v: 0.5*(grad(v) + grad(v).T)

    # Define equation F((u, p), (v, q)) = 0
    F = (2.0*eta*inner(strain(u), strain(v))*dx
         + div(v)*p*dx
         + div(u)*q*dx
         + H*inner(g, v)*dx)

    return lhs(F)

def form_action(T):
    """This function computes F(T)."""
    F = form(T)
    return assemble(action(F, w))

def derivative_action(T, dT):
    """This function computes dF/dT . dT."""
    F = action(form(T), w)
    deriv = derivative(F, T, dT)
    return assemble(deriv)

def convergence_order(errors):
    import math

    orders = [0.0] * (len(errors)-1)
    for i in range(len(errors)-1):
        try:
            orders[i] = math.log(errors[i]/errors[i+1], 2)
        except ZeroDivisionError:
            orders[i] = numpy.nan

    return orders

if __name__ == "__main__":
    # We're going to choose a perturbation direction, and then use that
    # direction 5 times, making the perturbation smaller each time.
    dT_dir = Function(V)
    dT_dir.vector()[:] = 1.0

    # We need the unperturbed F(T) to compare against.
    unperturbed = form_action(T)

    # fd_errors will contain
    # ||F(T+dT) - F(T)||
    fd_errors = []

    # grad_errors will contain
    # ||F(T+dT) - F(T) - dF/dT . dT||
    grad_errors = []

    # h is the perturbation size
    for h in [1.0e-5/2**i for i in range(5)]:
        # Build the perturbation
        dT = Function(V)
        dT.vector()[:] = h * dT_dir.vector()

        # Compute the perturbed result
        TdT = T.copy(deepcopy=True)
        TdT.vector()[:] += dT.vector()
        perturbed = form_action(TdT)

        fd_errors.append((perturbed - unperturbed).norm("l2"))
        grad_errors.append((perturbed - unperturbed - derivative_action(T, dT)).norm("l2"))

    # Now print the orders of convergence:
    print("Finite differencing errors: ", fd_errors)
    print("Finite difference convergence order (should be 1): ", convergence_order(fd_errors))
    print("Gradient errors: ", grad_errors)
    print("Gradient convergence order: ", convergence_order(grad_errors))
