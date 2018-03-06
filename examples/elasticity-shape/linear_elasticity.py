from dolfin import *
from dolfin_adjoint import *
from femorph import *
import matplotlib.pyplot as plt

"""Prepares 2D geometry. Returns facet function with 1, 2 on parts of
the boundary."""
n = 100
length = 2
x0 = 0.0
x1 = x0 + length
y0 = 0.0
y1 = 1.0
mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), int((x1-x0)*n), int((y1-y0)*n), 'crossed')
mesh = Mesh(mesh)
boundary_parts = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
left  = AutoSubDomain(lambda x: near(x[0], x0))
right = AutoSubDomain(lambda x: near(x[0], x1))
left .mark(boundary_parts, 1)
right.mark(boundary_parts, 2)


# Rotation rate and mass density
rho = 1.0
g = 9.81
# Loading due to gravity
f = Expression(("0", "-rho*g"),
               g=g, rho=rho, degree=2)

# Elasticity parameters
E = 1.0e3
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)
s = Function(V)
ALE.move(mesh, s)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v))*dx
L = inner(f, v)*dx

# Set up boundary condition on inner surface
c = Constant((0.0, 0.0))
bc = DirichletBC(V, c, boundary_parts, 1)

# Create solution function
u_fin = Function(V)
solve(a==L, u_fin, bcs=bc)

File("elasticity.pvd", "compressed") << u_fin

# Project and write stress field to post-processing file
# W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
# import dolfin
# stress = dolfin.project(sigma(u), V=W)
# File("stress.pvd") << stress

J = assemble(inner(u_fin,u_fin)*dx)
Jhat = ReducedFunctional(J, Control(s))
Jhat.optimize()
n = VolumeNormal(mesh)
bcs = DirichletBC(V, Constant((0.0,0.0)), boundary_parts, 1)
s2 = Function(V)
s2.interpolate(Expression(("x[0]","x[0]*x[1]"), degree=2))
bcs.apply(s2.vector())

print(J,Jhat(s2))
ALE.move(mesh, s2)

s0 = Function(V)
taylor_test(Jhat, s0, s2, dJdm=0)
print("-"*10)
taylor_test(Jhat, s0, s2)
