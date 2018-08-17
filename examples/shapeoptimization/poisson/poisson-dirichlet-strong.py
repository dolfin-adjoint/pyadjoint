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
a = inner(grad(u), grad(v))*dx
l = f*v*dx

g = Constant(1)
bc = DirichletBC(V, g, "on_boundary")
T = Function(V)
solve(a==l, T, bcs=bc)

T_file = File("output/T_strong.pvd")
T_file << T

J = assemble(T*T*dx)
c = Control(s)
Jhat = ReducedFunctional(J, c)

perturbation = project(Expression(("sin(x[0])*x[1]", "cos(x[1])"), degree=2), S)
taylor_test(Jhat, s, perturbation, dJdm=0)
taylor_test(Jhat, s, perturbation)

# Second order taylor
Jhat(s)
dJdm = Jhat.derivative().vector().inner(perturbation.vector())
Hm = compute_hessian(J, c, perturbation).vector().inner(perturbation.vector())
taylor_test(Jhat, s, perturbation, dJdm=dJdm, Hm=Hm)
