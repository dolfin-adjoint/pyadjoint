from dolfin import *
from femorph import *
import os
from dolfin_adjoint import *
from ufl import replace
import matplotlib.pyplot as plt
import numpy as np
set_log_level(LogLevel.ERROR)
L, H = 0.8,0.5
c = [L/2,H/2]
rot_c = [c[0]-0.1, c[1]]
rot_center = Point(rot_c[0],rot_c[1])
VariableBoundary = 1
FixedBoundary = 2
N, r = 50, 0.1

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

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="s")
ALE.move(mesh, s)

# Setup
V = FunctionSpace(mesh, "CG", 1)
guess_ic = Function(V, name="Initial Condition")
guess_ic.assign(Expression("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/H)", L=L, H=H, degree=1, name="IC Expression"))
u_prev = guess_ic.copy(deepcopy=True)
u_prev.rename("u^(t-1)","")
u_next = guess_ic.copy(deepcopy=True)
u_next.rename("u^t","")
half = Constant(0.5, name="0.5")
u_mid = half*u_prev + half*u_next
v = TestFunction(V)
states = [u_prev.copy(deepcopy=True)]
t = 0.0
dt = 0.0001
T = 100*dt
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
f = Expression("500*t/T*x[0]", t=t, T=T, degree=3)
while t < T:
    print("Solving for t == %.2e/%.2e" % (t + dt, T))
    F = inner((u_next - u_prev)/Constant(dt, name="dt"), v)*dx + inner(grad(u_mid), grad(v))*dx- f*v*dx
    solve(F == 0, u_next, J=derivative(F, u_next), annotate=True, bcs=bcs)

    ff << u_next
    u_prev.assign(u_next, annotate=True)
    t += dt
    f.t = t
    timestep += 1
    J += dt*assemble(u_next*dx)
    times.append(float(t))


Jhat = ReducedFunctional(J, Control(s))
tape.visualise("time_dependent_tape.dot", dot=True)

dJdmesh = Jhat.derivative()
bcs = DirichletBC(VectorFunctionSpace(mesh, "CG", 1), Constant((0,0)), marker, FixedBoundary)
bcs.apply(dJdmesh.vector())
File("ShapeGradient_time.pvd") << dJdmesh


n = VolumeNormal(mesh)
diff = Function(S)
diff.interpolate(Expression(("cos(6*pi*x[0])","cos(6*pi*x[0])"), degree=4))
s2 = Function(S)
print(Jhat(s))
s2.vector()[:] = 10*n.vector()[:]*diff.vector()[:]

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
