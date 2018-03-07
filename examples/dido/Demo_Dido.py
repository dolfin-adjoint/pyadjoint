from dolfin import *
from dolfin_adjoint import *
from femorph import *
import matplotlib.pyplot as plt

N = 10
mesh = Mesh(UnitSquareMesh(N, N))
One = Constant(1.0)

V = VectorFunctionSpace(mesh, "CG", 1)
s0 = Function(V)
ALE.move(mesh, s0)
print(assemble(One*dx(domain=mesh)))

alpha = 1
J = assemble(One*dx(domain=mesh)) + alpha*(assemble(One*ds(domain=mesh)) - 4)**2
Jhat = ReducedFunctional(J, Control(s0))
s = interpolate(Expression(("10*x[0]", "10*x[1]"), degree=2), V)
taylor_test(Jhat, s0, s, dJdm=0)
taylor_test(Jhat, s0, s)
