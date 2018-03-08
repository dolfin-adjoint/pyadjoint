from dolfin import *
import femorph
import matplotlib.pyplot as plt
import mshr
import numpy as np
import os

c = [0.5,0.5]
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

direction = femorph.VolumeNormal(mesh)
direction.vector()[:] *=-1.

f = Expression("x[0]*sin(x[0])*cos(x[1])", degree=4)

def StateEquation(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    n = FacetNormal(mesh)
    a = inner(grad(u), grad(v))*dx 
    L = inner(f,v)*dx

    # Assemble linear system
    A = assemble(a)
    b = assemble(L)
    marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
  
    bcs = [DirichletBC(V, Constant(0), marker, FixedBoundary),
           DirichletBC(V, Constant(1), marker, VariableBoundary)]
    [bc.apply(A,b) for bc in bcs]
    # Solving linear system
    T = Function(V, name="State")
    solve(A, T.vector(), b,'lu')
    return T


def AdjointEquation(T):
    mesh = T.function_space().mesh()
    V = T.function_space()
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = -inner(T,v)*dx
    
    A = assemble(a)
    b = assemble(L)
    marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
    bcs = [DirichletBC(V, Constant(0), marker, FixedBoundary)
           ,DirichletBC(V, Constant(0), marker, VariableBoundary)]
    [bc.apply(A,b) for bc in bcs]

    lmb = Function(V)
    solve(A, lmb.vector(), b, 'lu')
    return lmb

tf = File("output/T.pvd")

def phi(step):
    V = VectorFunctionSpace(mesh, "CG", 1)
    move = direction.copy(deepcopy=True)
    move.vector()[:]*=step

    u,v = TrialFunction(V), TestFunction(V)
    alpha = 1
    a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
    marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
    bcs =[ DirichletBC(V, Constant((0.,0.)), marker, FixedBoundary),
           DirichletBC(V, move, marker, VariableBoundary)]

    representation = Function(V)
    # b = step*inner(v,s)*ds
    b = inner(Constant((0.0,0.0)),v)*dx
    A = assemble(a)
    L = assemble(b)
    [bc.apply(A,L) for bc in bcs]
    solve(A, representation.vector(), L)
    ALE.move(mesh, representation)
    T = StateEquation(mesh)
    tf << T
    J = assemble(0.5*T*T*dx)
    revert = Function(V)
    revert.vector()[:] = - representation.vector()
    ALE.move(mesh, revert)
    # plot(representation)
    # plt.show()
    return(J)

T = StateEquation(mesh)
tf << T
lmb = AdjointEquation(T)
J_0 = assemble(0.5*T*T*dx)
print(J_0)
n = femorph.VolumeNormal(mesh)
marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
dS = Measure("ds", domain=mesh, subdomain_data=marker)
dJdm_paper=assemble(inner(direction, n)*(0.5*T*T-inner(n, grad(lmb))*inner(n, grad(T)))*dS(VariableBoundary))
print("gradient: ", dJdm_paper)
steps = np.array([0.25*r*0.5**i for i in range(7)])
err = np.zeros(len(steps))
j = 0
for step in steps:
    J_1 = phi(step)
    err[j] = J_1 - J_0
    j+=1
print(dJdm_paper, (J_1-J_0)/steps[0])
for i in range(len(steps)):
    print("gradient approx: ", err[i]/steps[i])
print("Rate1")
r_1 = np.log(np.abs(err[1:]/err[:-1]))/np.log(steps[1:]/steps[:-1])
print(r_1)
err_2_paper = np.zeros(len(steps))
for i in range(len(steps)):
    err_2_paper[i] = err[i]-steps[i]*dJdm_paper
r_2_paper = np.log(np.abs(err_2_paper[1:]/err_2_paper[:-1]))/np.log(steps[1:]/steps[:-1])
print("rate2paper")
print(r_2_paper)
print("Error2paper")
print(err_2_paper)
err_2 = np.zeros(len(steps))
dJdm=assemble(inner(direction, n)*0.5*T*T*dS(VariableBoundary))
for i in range(len(steps)):
    err_2[i] = err[i]-steps[i]*dJdm
r_2 = np.log(np.abs(err_2[1:]/err_2[:-1]))/np.log(steps[1:]/steps[:-1])
print("rate2")
print(r_2)
print("error2")
print(err_2)
