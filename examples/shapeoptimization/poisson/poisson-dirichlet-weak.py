from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(25,25)
S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S)
ALE.move(mesh, s)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)

f = 10*sin(pi*x[0])*sin(pi*x[1])
g = Constant(1)
n = FacetNormal(mesh)
alpha = 1e-2
a = inner(grad(u), grad(v))*dx
a_Nitsche = (-v*inner(grad(u), n) - u*inner(grad(v), n) + alpha*u*v)*ds
a += a_Nitsche
l = f*v*dx
l_Nitsche = (-Constant(1)*inner(grad(v), n) + alpha*g*v)*ds
l += l_Nitsche

T = Function(V)
solve(a==l, T)

T_file = File("output/T_weak.pvd")
T_file << T

J = assemble(T*T*dx)
Jhat = ReducedFunctional(J, Control(s))

perturbation = project(Expression(("sin(x[0])*x[1]", "cos(x[1])"), degree=2), S)
taylor_test(Jhat, s, perturbation, dJdm=0)
taylor_test(Jhat, s, perturbation)
