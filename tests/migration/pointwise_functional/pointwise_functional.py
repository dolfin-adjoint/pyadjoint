from fenics import *
from fenics_adjoint import *

from numpy.random import rand
from numpy.testing import assert_approx_equal

# Create mesh and define function space
mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

def center_func(x):
    return (0.45 <= x[0] and x[0] <= 0.55 and near(x[1], 0.5)) or \
           0.45 <= x[1] and x[1] <= 0.55 and near(x[0], 0.5)

# Define domain for point integral
center_domain = VertexFunction("size_t", mesh, 0)
center = AutoSubDomain(center_func)
center.mark(center_domain, 1)
dPP = dP(subdomain_data=center_domain)

# Define boundary condition
bc = DirichletBC(V, Constant(0), boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = project(Constant(0.4), V)
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u_ = Function(V)
solve(a == L, u_, bc)

# Compute the gradient of a functional
J = assemble(u_**2*dPP(1) + u_*dx)
Jhat = ReducedFunctional(J, Control(f))

assert_approx_equal(Jhat(f), J)

h = Function(V)
h.vector()[:] = rand(V.dim())
assert taylor_test(Jhat, f, h) > 1.9
