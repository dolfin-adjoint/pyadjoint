#!/usr/bin/env python
from fenics import *
from fenics_adjoint import *

mesh_xdmf = XDMFFile(mpi_comm_world(), "rectangle-less-circle.xdmf")
mesh = Mesh()
mesh_xdmf.read(mesh)

V_h = VectorElement("CG", mesh.ufl_cell(), 2)
Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V_h * Q_h)
V, Q = W.split()

v, q = TestFunctions(W)
x = TrialFunction(W)
u, p = split(x)
s = Function(W, name="State")
V_collapse = V.collapse()
g = Function(V_collapse, name="Control")

class Circle(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]-10)**2 + (x[1]-5)**2 < 3**2

facet_marker = FacetFunction("size_t", mesh)
facet_marker.set_all(10)
Circle().mark(facet_marker, 2)

ds = ds(subdomain_data=facet_marker)

# Set parameter values
nu = Constant(1)     # Viscosity coefficient
gamma = Constant(10)    # Nitsche penalty parameter
n = FacetNormal(mesh)
h = CellSize(mesh)

# Define boundary conditions
u_inflow = Expression(("x[1]*(10-x[1])/25", "0"), degree=1)
noslip = DirichletBC(W.sub(0), (0, 0),
                     "on_boundary && (x[1] >= 9.9 || x[1] < 0.1)")
inflow = DirichletBC(W.sub(0), u_inflow, "on_boundary && x[0] <= 0.1")
circle = DirichletBC(W.sub(0), g, facet_marker, 2)
bcs = [inflow, noslip, circle]

a = (nu*inner(grad(u), grad(v))*dx
     - inner(p, div(v))*dx
     - inner(q, div(u))*dx
     )
f = Function(V.collapse())
L = inner(f, v)*dx

A, b = assemble_system(a, L, bcs)
solve(A, s.vector(), b)

u, p = split(s)
alpha = Constant(10)

J = assemble(1./2*inner(u, u)**2*dx)
dJdm = compute_gradient(J, f)
