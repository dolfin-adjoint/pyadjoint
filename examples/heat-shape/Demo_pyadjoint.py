from dolfin import *
from femorph import *
import mshr
import os
from dolfin_adjoint import *
from ufl import replace
import matplotlib.pyplot as plt
import numpy as np

c = [0.3,0.5]
rot_c = [c[0]-0.1, c[1]]
rot_center = Point(rot_c[0],rot_c[1])
VariableBoundary = 2
FixedBoundary = 1
N, r = 50, 0.05
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
    data[55] = "bc = %s;\n" %int(FixedBoundary)
    data[56] = "object = %s;\n" %int(VariableBoundary)
    with open("mesh.geo", 'w') as file:
        file.writelines( data )
os.system("gmsh -2 mesh.geo -o mesh.msh")
os.system("dolfin-convert mesh.msh mesh.xml")

mesh =  Mesh("mesh.xml")

f = Expression("x[0]*sin(x[0])*cos(x[1])", degree=4)

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S)
ALE.move(mesh, s)

# Setup
V = FunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)
F = inner(grad(u), grad(v))*dx  - f*v*dx + u*v*dx

# FIXME: Handling of Dirichlet Conditions
marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
bcs = [DirichletBC(V, Constant(0), marker, FixedBoundary)]#,
       # DirichletBC(V, Constant(1), marker, VariableBoundary)]


T = Function(V, name="T")

solve(lhs(F) == rhs(F), T, bcs=bcs)

J = 0.5*T*T*dx
J = assemble(J)
Jhat = ReducedFunctional(J, Control(s))

n = VolumeNormal(mesh)
s2 = Function(S)
BC = DirichletBC(S, Constant((0,0)), marker, FixedBoundary)
ALE.move(mesh,s2)
plot(mesh)
plt.show()
s2.vector()[:] = -2*n.vector()[:]
s0 = Function(S)

taylor_test(Jhat, s0, s2, dJdm=0)

print("-"*10)
taylor_test(Jhat, s0, s2)

