from fenics import *
from fenics_adjoint import *

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 1)

# Set up controls
a = Constant(0)
b = Constant(1)
ctrls = [Control(a), Control(b)]

# Evaluate forward model and functional. Note that only b is being used - and c
# never show up
x = project(b, V, annotate=True)
J = assemble(x**2*dx)

# Compute gradient
dJ = compute_gradient(J, ctrls)
assert dJ[0] is None
assert dJ[1] is not None

# For zero derivatives, return a dolfin object rather None
# TODO: Is this something to consider for pyadjoint?
parameters["adjoint"]["allow_zero_derivatives"] = True
dJ = compute_gradient(J, ctrls)
assert dJ[0] is not None
assert dJ[1] is not None
assert float(dJ[0]) == 0.0

