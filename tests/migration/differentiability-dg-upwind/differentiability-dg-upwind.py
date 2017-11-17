"""This script implements the Taylor remainder convergence test
for an individual form.

Imagine we have an expression F(u) that is a function of velocity. We
can check the correctness of the derivative dF/du by noting that

||F(u + du) - F(u)|| converges at first order

but that

||F(u + du) - F(u) - dF/du . du|| converges at second order.

In this example, F(u) is the action of the advection operator
on a supplied temperature field:

F(u) = action(advection(u), T).

"""

from __future__ import print_function
from numpy import random
from dolfin import *

mesh = Mesh("mesh.xml.gz")

p2 = VectorElement("CG", triangle, 2)
p1 = FiniteElement("CG", triangle, 1)
p2p1 = MixedElement([p2, p1])

V = FunctionSpace(mesh, p2)
W = FunctionSpace(mesh, p2p1)
V = FunctionSpace(mesh, "DG", 1)

w = Function(W, "velocity.xml.gz")
T = Function(V, "temperature.xml.gz")

u_pvd = File("velocity.pvd")
u_pvd << w.split()[0]
t_pvd = File("temperature.pvd")
t_pvd << T

def form(w):
    T = TrialFunction(V)
    v = TestFunction(V)

    h = CellSize(mesh)
    n = FacetNormal(mesh)

    u = split(w)[0]
    un = abs(dot(u('+'), n('+')))
    jump_v = v('+')*n('+') + v('-')*n('-')
    jump_T = T('+')*n('+') + T('-')*n('-')

    F = -dot(u*T, grad(v))*dx + (dot(u('+'), jump_v)*avg(T))*dS + dot(v, dot(u, n)*T)*ds + 0.5*un*dot(jump_T, jump_v)*dS
    return F

def form_action(w):
    """This function computes F(u)."""
    F = form(w)
    return assemble(action(F, T))

def derivative_action(w, dw):
    """This function computes dF/du . du."""
    F = action(form(w), T)
    deriv = derivative(F, w, dw)
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
    # We're going to choose a random perturbation direction, and then use that
    # direction 5 times, making the perturbation smaller each time.
    dw_dir = Function(W)
    dw_dir.vector()[:] = random.random((W.dim(),))

    # We need the unperturbed F(u) to compare against.
    unperturbed = form_action(w)

    # fd_errors will contain
    # ||F(u+du) - F(u)||
    fd_errors = []

    # grad_errors will contain
    # ||F(u+du) - F(u) - dF/du . du||
    grad_errors = []

    # h is the perturbation size
    for h in [0.1/2**i for i in range(5)]:
        # Build the perturbation
        dw = Function(W)
        dw.vector()[:] = h * dw_dir.vector()

        # Compute the perturbed result
        wdw = w.copy(deepcopy=True) # w + dw
        wdw.vector()[:] += dw.vector()
        perturbed = form_action(wdw)

        fd_errors.append((perturbed - unperturbed).norm("l2"))
        grad_errors.append((perturbed - unperturbed - derivative_action(w, dw)).norm("l2"))

    # Now print the orders of convergence:
    print("Finite differencing errors: ", fd_errors)
    print("Finite difference convergence order (should be 1): ", convergence_order(fd_errors))
    print("Gradient errors: ", grad_errors)
    print("Gradient convergence order: ", convergence_order(grad_errors))
