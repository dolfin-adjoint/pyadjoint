from dolfin import *
from dolfin_adjoint import *

mesh = Mesh("meshes/mesh.xml")
mf = MeshFunction("size_t", mesh, "meshes/mesh_facet_region.xml")
obstacle = 2
boundary = 3

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S)
ALE.move(mesh, s)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)

a = inner(grad(u), grad(v))*dx + u*v*dx

g = Constant(1)
bc = DirichletBC(V, g, mf, 2)
T = Function(V)
solve(lhs(a)==rhs(a), T, bcs=bc)

T_file = File("output/T_time_indep.pvd")
T_file << T

dB = Measure("ds", domain=mesh, subdomain_data=mf, subdomain_id=boundary)
J = assemble(T*T*dB)
Jhat = ReducedFunctional(J, Control(s))

def smooth_representation(value):
    u,v = TrialFunction(S), TestFunction(S)
    alpha = 1
    a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
    bc_b = DirichletBC(S, Constant((0,0)), mf, boundary) 
    bc_o = DirichletBC(S, value, mf, obstacle)
    representation = Function(S)
    solve(lhs(a)==rhs(a), representation, bcs=[bc_b, bc_o])
    return representation

perturbation = project(Expression(("sin(x[0])*x[1]", "cos(x[1])"), degree=2), S)
perturbation = smooth_representation(perturbation)

taylor_test(Jhat, s, perturbation, dJdm=0)
taylor_test(Jhat, s, perturbation)

