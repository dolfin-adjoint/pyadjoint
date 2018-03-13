from dolfin import *
from femorph import *
import mshr
import os
from dolfin_adjoint import *
from ufl import replace
import matplotlib.pyplot as plt
import numpy as np

c = [0.5,0.5]
rot_c = [c[0]-0.1, c[1]]
rot_center = Point(rot_c[0],rot_c[1])
VariableBoundary = 1
FixedBoundary = 2
N, r = 100, 0.2
L, H = 1,1

#neumann = False
neumann = True
with open("mesh.geo", 'r') as file:
    data = file.readlines()
    data[0] = "lc1 = %s;\n" %(float(1/N))
    data[2] = "cx = %s;\n" %(float(c[0]))
    data[3] = "cy = %s;\n" %(float(c[1]))
    data[4] = "a = %s;\n" %(float(r))
    data[5] = "b = %s;\n" %(float(2*r))

    data[7] = "L = %s;\n" %(float(L))
    data[8] = "H = %s;\n" %(float(H))
    data[54] = "bc = %s;\n" %int(FixedBoundary)
    data[55] = "object = %s;\n" %int(VariableBoundary)
    with open("mesh.geo", 'w') as file:
        file.writelines( data )
os.system("gmsh -2 mesh.geo -o mesh.msh")
os.system("dolfin-convert mesh.msh mesh.xml")

mesh =  Mesh("mesh.xml")

f = Expression("100*x[0]*sin(x[0])*cos(x[1])", degree=4, name="f")

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="s")
ALE.move(mesh, s)

# Setup
V = FunctionSpace(mesh, "CG", 1)
guess_ic = Function(V, name="Initial Condition")
guess_ic.assign(Expression("15 * x[0] * (1 - x[0]) * x[1] * (1 - x[1])", degree=1, name="IC Expression"))
u_prev = guess_ic.copy(deepcopy=True)
u_prev.rename("u^(t-1)","")
u_next = guess_ic.copy(deepcopy=True)
u_next.rename("u^t","")
half = Constant(0.5, name="0.5")
u_mid = half*u_prev + half*u_next
v = TestFunction(V)
states = [u_prev.copy(deepcopy=True)]
t = 0.0
dt = 0.01
T = 2*dt
times  = [float(t)]
timestep = 0
ff = File("output/Demo_timedependent.pvd")

marker = MeshFunction("size_t", mesh, "mesh_facet_region.xml")
if neumann:
    bcs = [DirichletBC(V, Constant(0,name="Dirichlet Condition"), marker, FixedBoundary)]
else:
    bcs = [DirichletBC(V, Constant(0), marker, FixedBoundary),
           DirichletBC(V, Constant(1), marker, VariableBoundary)]
J = 0
while t < T:
    print("Solving for t == %s" % (t + dt))
    F = inner((u_next - u_prev)/Constant(dt, name="dt"), v)*dx + inner(grad(u_mid), grad(v))*dx
    solve(F == 0, u_next, J=derivative(F, u_next), annotate=True, bcs=bcs)

    ff << u_next
    u_prev.assign(u_next, annotate=True)
    t += dt
    timestep += 1
    J += assemble(u_next*dx)
    times.append(float(t))


Jhat = ReducedFunctional(J, Control(s))
tape.visualise("time_dependent_tape.dot", dot=True)

dJdmesh = Jhat.derivative()
bcs = DirichletBC(VectorFunctionSpace(mesh, "CG", 1), Constant((0,0)), marker, FixedBoundary)
bcs.apply(dJdmesh.vector())
File("ShapeGradient_time.pvd") << dJdmesh


n = VolumeNormal(mesh)
s2 = Function(S)
print(Jhat(s))
s2.vector()[:] = 10*n.vector()[:]
bcs = DirichletBC(VectorFunctionSpace(mesh, "CG", 1), Constant((0,0)), marker, FixedBoundary)
bcs.apply(s2.vector())

dJdm = Jhat.derivative()
step = 15/N
boundary_move = s2.copy(deepcopy=True)
boundary_move.vector()[:] *= step

bcs = [DirichletBC(S, Constant((0,0)), marker, FixedBoundary),
       DirichletBC(S, boundary_move, marker, VariableBoundary)]
u,v = TrialFunction(S), TestFunction(S)
F = inner(grad(u), grad(v))*dx+inner(u,v)*dx
s3 = Function(S)
solve(lhs(F)==rhs(F), s3, bcs=bcs, annotate=False)

s0 = Function(S)
print("-"*10)
taylor_test(Jhat, s0, s3, dJdm=0)
taylor_test(Jhat, s0, s3)
s3.vector()[:] *= 0.01
movement = File("movement.pvd") 
for i in range(4):
    movement << s3
    s3.vector()[:]*=0.5
