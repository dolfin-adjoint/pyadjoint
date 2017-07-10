from fenics import *
from fenics_adjoint import *
from normalise_overloaded import normalise

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'CG', 1)

f = project(Expression('x[0]*x[1]', degree=1), V)

g = normalise(f)

J = assemble(g*dx)

h = Function(V)
h.vector()[:] = 0.1
taylor_test(ReducedFunctional(J, f), f, h)
