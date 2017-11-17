from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
f = interpolate(Expression("sin(pi*x[0])", degree=1), V)
f = Function(V, f.vector(), name="ic")

def main(f, annotate=False):
    u = Function(V)
    a = u*v*dx
    L = f*v*dx
    F = a - L

    bcs = None

    problem = NonlinearVariationalProblem(F, u, bcs, J=derivative(F, u))
    solver = NonlinearVariationalSolver(problem)
    solver.solve(annotate=annotate)
    return u

u = main(f, annotate=True)
if False:
    # TODO: Not implemented.
    assert replay_dolfin()

grad = compute_gradient(assemble(u*u*dx +
                        f*f*dx), Control(f))
h = Function(V)
h.vector()[:] = rand(V.dim())
grad = h._ad_dot(grad)

def J(f):
    u = main(f, annotate=False)
    return assemble(u*u*dx + f*f*dx)

minconv = taylor_test(J, f, h, grad)
assert minconv > 1.9
