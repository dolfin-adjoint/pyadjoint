import numpy as np
from dolfin import *
from dolfin_adjoint import *

tape = Tape()
set_working_tape(tape)
mesh = UnitSquareMesh(6, 6)
X = SpatialCoordinate(mesh)
S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S,name="deform")
ALE.move(mesh, s)
f = X[0]*X[1]
V = FunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v))*dx
l = f*v*dx
bc = DirichletBC(V, Constant(1), "on_boundary")
u = Function(V)
solve(a==l, u, bcs=bc)

J = assemble(u*dx(domain=mesh))**2
c = Control(s)
Jhat = ReducedFunctional(J, c)

f = Function(S, name="W")
f.interpolate(Expression(("A*sin(x[1])", "A*cos(x[1])"),degree=2,A=10))
h = Function(S,name="V")
h.interpolate(Expression(("A*cos(x[1])", "A*x[1]"),degree=2,A=10))
tape.visualise("hessian_solve.dot", dot=True)

# Finite difference
r0 = taylor_test(Jhat, s, h, dJdm=0)
Jhat(s)
assert(r0>0.95)
# First order taylor
s.tlm_value = h
tape.evaluate_tlm()
r1 = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
assert(r1>1.95)
Jhat(s)

# Second order taylor
dJdm = Jhat.derivative().vector().inner(h.vector())
Hm = compute_hessian(J, c, h).vector().inner(h.vector())
taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
Jhat(s)
# dJdmm_exact = derivative(derivative(sin(X[1])* dx(domain=mesh),X,h), X, h)


# # ---------------- hessian for assembly ----------------------
# import numpy as np
# from dolfin import *
# from dolfin_adjoint import *

# tape = Tape()
# set_working_tape(tape)
# mesh = UnitSquareMesh(6, 6)
# X = SpatialCoordinate(mesh)
# S = VectorFunctionSpace(mesh, "CG", 1)
# s = Function(S,name="deform")

# ALE.move(mesh, s)
# J = assemble(sin(X[1])* dx(domain=mesh))
# c = Control(s)
# Jhat = ReducedFunctional(J, c)

# f = Function(S, name="W")
# f.interpolate(Expression(("A*sin(x[1])", "A*cos(x[1])"),degree=2,A=10))
# h = Function(S,name="V")
# h.interpolate(Expression(("A*cos(x[1])", "A*x[1]"),degree=2,A=10))


# # Finite difference
# taylor_test(Jhat, s, h, dJdm=0)
# Jhat(s)

# # First order taylor
# s.tlm_value = h
# tape.evaluate_tlm()
# taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
# Jhat(s)

# # Second order taylor
# dJdm = Jhat.derivative().vector().inner(h.vector())
# Hm = compute_hessian(J, c, h).vector().inner(h.vector())
# taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
# Jhat(s)
# dJdmm_exact = derivative(derivative(sin(X[1])* dx(domain=mesh),X,h), X, h)

