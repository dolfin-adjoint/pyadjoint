from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
f = interpolate(Expression("sin(pi*x[0])", degree=1), V)
# TODO: Workaround until interpolate is overloaded.
f = Function(V, f.vector())

def main(f, annotate=False):
    u = Function(V, name = "system state")
    w = TrialFunction(V)
    a = w*v*dx
    L = f*v*dx
    F = a - L

    bcs = None

    problem = LinearVariationalProblem(a, L, u, bcs)
    problem = LinearVariationalProblem(a, L, u)
    solver = LinearVariationalSolver(problem)
    solver.solve(annotate=annotate)

    return u

u = main(f, annotate=True)
if False:
    # TODO: Not implemented yet.
    assert replay_dolfin()

grad = compute_gradient(assemble(u*u*dx), Control(f))
h = Function(V)
h.vector()[:] = rand(V.dim())
grad = h._ad_dot(grad)

def J(f):
    u = main(f, annotate=False)
    return assemble(u*u*dx)

minconv = taylor_test(J, f, h, grad)
assert minconv > 1.9
