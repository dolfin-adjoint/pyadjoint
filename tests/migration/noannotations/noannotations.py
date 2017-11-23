from fenics import *
from fenics_adjoint import *

mesh = UnitSquareMesh(4, 4)
V0 = FunctionSpace(mesh, "CG", 1)

u0 = project(Constant(1), V0, annotate=False)
u0.assign(10*u0, annotate=False)

tape = get_working_tape()
assert len(tape.get_blocks()) == 0
