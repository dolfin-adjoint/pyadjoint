from dolfin import *
from femorph import *
import mshr
import os
from ufl import replace
import matplotlib.pyplot as plt
import numpy as np

c = [0.5,0.5]
rot_c = [c[0]-0.1, c[1]]
rot_center = Point(rot_c[0],rot_c[1])
VariableBoundary = 2
FixedBoundary = 1
N, r = 50, 0.1
L, H = 1,1
with open("mesh.geo", 'r') as file:
    data = file.readlines()
    data[0] = "lc1 = %s;\n" %(float(1/N))
    data[2] = "cx = %s;\n" %(float(c[0]))
    data[3] = "cy = %s;\n" %(float(c[1]))
    data[4] = "a = %s;\n" %(float(r))
    data[5] = "b = %s;\n" %(float(3*r))

    data[7] = "L = %s;\n" %(float(L))
    data[8] = "H = %s;\n" %(float(H))
    data[54] = "bc = %s;\n" %int(FixedBoundary)
    data[55] = "object = %s;\n" %int(VariableBoundary)
    with open("mesh.geo", 'w') as file:
        file.writelines( data )
os.system("gmsh -2 mesh.geo -o mesh.msh")
os.system("dolfin-convert mesh.msh mesh.xml")

mesh =  Mesh("mesh.xml")
f = Expression("100*x[0]*sin(x[0])*cos(x[1])", degree=4)

def solve_state(mesh):
    S = VectorFunctionSpace(mesh, "CG", 1)

    # Setup
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    F = inner(grad(u), grad(v))*dx  - f*v*dx # + u*v*dx

    # FIXME: Handling of Dirichlet Conditions
    marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
    bcs = [DirichletBC(V, Constant(0), marker, FixedBoundary),
           DirichletBC(V, Constant(1), marker, VariableBoundary)]


    T = Function(V, name="T")
    
    solve(lhs(F) == rhs(F), T, bcs=bcs)
    return T
    
def solve_adjoint(T):
    V = T.function_space()
    mesh = V.mesh()
    u, v = TrialFunction(V), TestFunction(V)
    F = inner(grad(u),grad(v))*dx + inner(T, v)*dx
    marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
    bcs = [DirichletBC(V, Constant(0), marker, FixedBoundary),
           DirichletBC(V, Constant(0), marker, VariableBoundary)]
    lmb = Function(V, name="lmb")
    solve(lhs(F) == rhs(F), lmb, bcs=bcs)
    return lmb

T = solve_state(mesh)
lmb = solve_adjoint(T)

J_0 = assemble(0.5*T*T*dx)
print(J_0)
marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
dNeumann = Measure("ds", domain=mesh, subdomain_data = marker)

S = VectorFunctionSpace(mesh, "CG", 1)
n, s = VolumeNormal(mesh), TestFunction(S)
dJds = assemble(inner(n,s)*(0.5*T*T )*dNeumann(VariableBoundary))
dFds = assemble(inner(n,s)*( - lmb*f + inner(grad(lmb), grad(T)))*dNeumann(VariableBoundary))
dFdDir = assemble(inner(n,s)*(-inner(n,grad(lmb))*inner(n, grad(T)))*dNeumann(VariableBoundary))

step = 5/N
n = VolumeNormal(mesh)
s2 = Function(S)
s2.vector()[:] = -0.5*n.vector()[:]
bcs = DirichletBC(VectorFunctionSpace(mesh, "CG", 1), Constant((0,0)), marker, FixedBoundary)
bcs.apply(s2.vector())
boundary_move = s2.copy(deepcopy=True)

boundary_move.vector()[:] *= step
dJdboundary = dJds.inner(boundary_move.vector())
dFdboundary = dFds.inner(boundary_move.vector())
dDirdboundary = dFdDir.inner(boundary_move.vector())

plot(mesh)
marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")

ALE.move(mesh, boundary_move)
plot(mesh, color="r")
T_step = solve_state(mesh)


# Smoothed version
J_1 = assemble(0.5*T_step*T_step*dx)
revert = boundary_move.copy(deepcopy=True)
revert.vector()[:] *= -1
ALE.move(mesh, revert)

print(dJdboundary, J_1-J_0, J_1)
print(dDirdboundary)
print(dJdboundary+dDirdboundary, J_1-J_0)
print("this is promising")
bcs = [DirichletBC(S, Constant((0,0)), marker, FixedBoundary),
       DirichletBC(S, boundary_move, marker, VariableBoundary)]
u,v = TrialFunction(S), TestFunction(S)
F = inner(grad(u), grad(v))*dx + inner(u,v)*dx
s3 = Function(S)
solve(lhs(F)==rhs(F), s3, bcs=bcs)
ALE.move(mesh, s3)
plot(mesh, color="g")
plt.show()

T_step = solve_state(mesh)
J_1 = assemble(0.5*T_step*T_step*dx)
print(dJdboundary+dDirdboundary, J_1-J_0, J_1)
