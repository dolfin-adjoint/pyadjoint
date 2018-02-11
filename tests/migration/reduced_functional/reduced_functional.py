from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitIntervalMesh(mpi_comm_world(), 2)

W = FunctionSpace(mesh, "CG", 1)
rho = project(Constant(1), W)
g = project(Constant(1), W)

u = Function(W, name="State")
u_ = TrialFunction(W)
v = TestFunction(W)

F = rho*u_*v * dx - Constant(1)*g*v*dx
solve(lhs(F) == rhs(F), u)

J = assemble(0.5 * inner(u, u) * dx + g**3*dx)

# Reduced functional with single control
m = Control(rho)

Jhat = ReducedFunctional(J, m)
Jhat.derivative()
Jhat(rho)

if False:
    # TODO: This is not implemented. And exactly how the interface should be has not been decided.
    Jhat.hessian(rho)

h = Function(W)
h.vector()[:] = rand(W.dim())
assert taylor_test(Jhat, rho, h) > 1.9


# Reduced functional with multiple controls
m2 = Control(g)

Jhat = ReducedFunctional(J, [m, m2])
Jhat.derivative()
Jhat([rho, g])
if False:
    # TODO: This is not implemented. And exactly how the interface should be has not been decided.
    Jhat.hessian([rho, g])

h2 = Function(W)
h2.vector()[:] = rand(W.dim())
assert taylor_test(Jhat, [rho, g], [h, h2]) > 1.9
