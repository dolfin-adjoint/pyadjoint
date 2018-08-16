import numpy as np
from dolfin import *
from dolfin_adjoint import *

#tape = Tape()
#set_working_tape(tape)
mesh = UnitSquareMesh(6, 6)
X = SpatialCoordinate(mesh)
S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S,name="deform")

ALE.move(mesh, s)
J = assemble(sin(X[1])* dx(domain=mesh))
c = Control(s)
Jhat = ReducedFunctional(J, c)

f = Function(S)
f.vector()[:] = 2

#tape.evaluate_tlm()
h = Function(S,name="random")
h.interpolate(Expression(("10*x[0]*cos(x[1])", "10*x[1]"),degree=2))


taylor_test(Jhat, s, h, dJdm=0)

taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)

dJdm = Jhat.derivative().vector().inner(h.vector())
Hm = compute_hessian(J, c, h).vector().inner(h.vector())
assert(taylor_test(Jhat, f, h, dJdm=dJdm, Hm=Hm) > 2.9)
