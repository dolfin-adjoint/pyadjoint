from fenics import *
from fenics_adjoint import *
import numpy as np

mesh = UnitSquareMesh(2,2)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
R = FunctionSpace(mesh, "R", 0)

a = interpolate(Constant(1.0), R, name="a")
b = interpolate(Constant(2.0), R, name="b")

f = a*u*v*dx
l = b*v*dx

u = Function(V)

F = assemble(f)
L = assemble(l)
solve(F, u.vector(), L)

J = Functional(u*dx)

#Test the hessian array
test = 0

def hessian_cb(j, m, mdot, h):
    global test
    test = 1

J_hat = ReducedFunctional(J, Control(a), hessian_cb = hessian_cb)
J_hat(a)
J_hat.hessian(a)
assert test == 1, "Hessian callback was not called."
