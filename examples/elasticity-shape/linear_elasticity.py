from dolfin import *
from dolfin_adjoint import *
from femorph import *
import matplotlib.pyplot as plt
import numpy as np
"""Prepares 2D geometry. Returns facet function with 1, 2 on parts of
the boundary."""
n = 50
length = 2
x0 = 0.0
x1 = x0 + length
height = 0.2
y0 = 0.0
y1 = y0 + height

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
bcLeft = DirichletBC(V, c, boundary_parts, 1)
bcRight = DirichletBC(V, c, boundary_parts, 2)
# Create solution function
u_fin = Function(V, name="deform")
solve(a==L, u_fin, bcs=[bcLeft])
#solve(a==L, u_fin, bcs=[bcLeft, bcRight])

File("elasticity.pvd", "compressed") << u_fin

# Project and write stress field to post-processing file
# W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
# import dolfin
# stress = dolfin.project(sigma(u), V=W)
# File("stress.pvd") << stress
alpha_reg = 1e2
alpha_vol = 1e1
one = interpolate(Constant(1.0), FunctionSpace(mesh, "CG", 1))
J = assemble(inner(u_fin,u_fin)*dx)
J += alpha_reg*assemble(inner(s,s)*dx)
J += alpha_vol*(assemble(one*dx)-height*length)*(assemble(one*dx)-height*length)
Jhat = ReducedFunctional(J, Control(s))
Jhat.optimize()

n = VolumeNormal(mesh)
s2 = Function(V)
# s2.interpolate(Expression(("0","0.02*x[0]*x[0]*(length-x[0])*(length-x[0])"), length=length, degree=2))
#[bc.apply(s2.vector()) for bc in [bcLeft, bcRight]]
s2.interpolate(Expression(("5*x[0]","5*x[0]*x[1]"), x0=x0, degree=2))
[bc.apply(s2.vector()) for bc in [bcLeft]]
s0 = Function(V)
taylor_test(Jhat, s0, s2, dJdm=0)
print("-"*10)
taylor_test(Jhat, s0, s2)

def riesz_representation(value):
    u,v = TrialFunction(V), TestFunction(V)
    alpha = 1
    a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
    A = assemble(a)
    # [bc.apply(A,value) for bc in [bcLeft, bcRight]]
    [bc.apply(A,value) for bc in [bcLeft]]

    representation = Function(V)
    solve(A, representation.vector(), value)
    return representation


it,max_it = 0, 50
min_stp = 1e-6
meshout = File("output/mesh.pvd")
meshout << mesh
Js = []

import time
move = Function(V)
while it <= max_it:
    print("-"*5, "Iteration %d" %it, "-"*5)
    if it == 0:
        J_step = Jhat(s)
        print(J_step)
    Js.append(J_step)
    meshout << mesh
    dJdm = Jhat.derivative(options={"riesz_representation":
                                    riesz_representation})
    step =  1e-1
    while (step>min_stp):
        dJdm_step = Function(V)
        dJdm_step.vector()[:] = -step*dJdm.vector()
        move.vector()[:] += step*dJdm_step.vector()
        J_step = Jhat(move)
        if J_step-Js[it]<=0:
            print("Decreasing direction found")
            print((J_step-Js[it])/J_step)
            if np.abs(J_step-Js[it])/np.abs(J_step)<=1e-6:
                it = max_it + 1
                print("Relative reduction minima")
            step = min_stp
        else:
            # Reset mesh and half step-size
            move.vector()[:] -= dJdm_step.vector()
            step /=2
            if step<=min_stp:
                print("Minimum step-length reached")
                it = max_it+1

    it+=1


