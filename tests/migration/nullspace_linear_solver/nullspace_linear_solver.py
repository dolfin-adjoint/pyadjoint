""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2

    subjecct to

    grad \cdot \grad u = f    in \Omega
    du/dn = 0                 on \partial \Omega


"""
from dolfin import *
from dolfin_adjoint import *

plotting = False

set_log_level(ERROR)

# Create mesh, refined in the center
n = 8
mesh = UnitSquareMesh(n, n)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = interpolate(Expression("x[0]", degree=1), W, name='Control')
u = Function(V, name='State')
v = TrialFunction(V)
w = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
a = inner(grad(v), grad(w)) * dx
L = f * w * dx
A = assemble(a)
b = assemble(L)

# System Ax = b is singular -> set nullspace
constants = Function(V).vector()
constants[:] = 1; constants /= constants.norm("l2")
nullspace = VectorSpaceBasis([constants])
#nullspace.orthogonalize(b)
as_backend_type(A).set_nullspace(nullspace)

# now we can solve
solver = LinearSolver(mpi_comm_world(), "default")
solver.set_nullspace(nullspace)
solver.solve(A, u.vector(), b)


# Define functional of interest and the reduced functional
x = SpatialCoordinate(mesh)
d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile

alpha = Constant(1e-4)
J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = Control(f)
rf = ReducedFunctional(J, control)
f_opt = minimize(rf, tol=1.0e-10)

# Check that the functional is small enough. If that is not the case, we
# nullspace handling is probably broken
j_opt = rf(f_opt)
assert j_opt < 1e-3

if plotting:
    plot(f_opt)

    # find optimal state
    b = assemble(f_opt * w * dx, PETScVector())
    nullspace.orthogonalize(b)
    solver.solve(A, u.vector(), b)
    d = interpolate(Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])", degree=1), V)
    nullspace.orthogonalize(d.vector())
    plot(d)
    plot(u, interactive = True)
