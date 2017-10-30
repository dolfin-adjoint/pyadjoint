from fenics import *
from fenics_adjoint import *

mesh = UnitIntervalMesh(mpi_comm_world(), 2)

W = FunctionSpace(mesh, "CG", 1)
rho = Constant(1)
g = Constant(1)


u = Function(W, name="State")
u_ = TrialFunction(W)
v = TestFunction(W)

F = rho*u_*v * dx - g*v*dx(domain=mesh)
solve(lhs(F) == rhs(F), u)

J = assemble(0.5 * inner(u, u) * dx + g**3*dx(domain=mesh))

# Reduced functional with single control
m = Control(rho)

Jhat = ReducedFunctional(J, m)
Jhat.derivative()
Jhat(rho)

h = Constant(1.0)
assert taylor_test(Jhat, rho, h) > 1.9

# Reduced functional with multiple controls
m2 = Control(g)

Jhat = ReducedFunctional(J, [m, m2])
Jhat.derivative()
Jhat([rho, g])

direction = [Constant(1), Constant(1)]

assert taylor_test_multiple(Jhat, [rho, g], direction) > 1.9
