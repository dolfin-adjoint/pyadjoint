from fenics import *
from fenics_adjoint import *
from normalise_overloaded import normalise

from numpy.random import rand

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'CG', 1)

f = project(Expression('x[0]*x[1]', degree=1), V)

g = normalise(f)

J = assemble(g*dx)

h = Function(V)
h.vector()[:] = rand(h.vector().local_size())

taylor_test(ReducedFunctional(J, Control(f)), f, h)
