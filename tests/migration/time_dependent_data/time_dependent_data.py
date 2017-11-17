'''Contributed by Martin Alnaes, launchpad question 228870'''

from fenics import *
from fenics_adjoint import *

mesh = UnitSquareMesh(5,5)
V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)

z = Function(R, name="z")

ut = TrialFunction(V)
v = TestFunction(V)
m = Function(V, name="m")
u = Function(V, name="u")

nt = 3
J = 0
for t in range(nt):
    tmp = interpolate(Expression("t*t", t=t, degree=2), R) # ... to make sure this is recorded
    tmp = Function(R, tmp.vector())
    z.assign(tmp)
    solve(ut*v*dx == m*v*dx, u, []) # ... even though it's not used in the computation
    if t == nt-1:
        quad_weight = AdjFloat(0.5)
    else:
        quad_weight = AdjFloat(1.0)
    J += quad_weight * assemble((u-z)**2*dx)

# ... so it can be replayed by the functional at the end
J = ReducedFunctional(J, Control(m))
assert abs(float(J(m)) - 9.0) < 1.0e-14
