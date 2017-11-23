from fenics import *
from fenics_adjoint import *

# This test checks that passing in solver parameters as
# a dolfin.Parameters object is handled in dolfin-adjoint
params = NonlinearVariationalSolver.default_parameters()

# Setup
n = 200
mesh = RectangleMesh(mpi_comm_world(), Point(-1, -1), Point(1, 1), n, n)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V, name="State")
m = Function(V, name="Control")
v = TestFunction(V)

m.interpolate(Constant(0.05))

# Run the forward model once to create the simulation record
F = (inner(grad(u), grad(v)) - m*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc, solver_parameters=params)
