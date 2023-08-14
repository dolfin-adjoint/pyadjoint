from dolfin import *
from dolfin_adjoint import *

import numpy as np

# Define mesh
mesh = UnitSquareMesh(10, 10)
n    = FacetNormal(mesh)
U    = FunctionSpace(mesh, "CG", 1)

adj_start_timestep()

def forward(c):
    u  = Function(U, name="state")
    u0 = Function(U, name="oldstate")
    v  = TestFunction(U)
    F  = inner(u - u0 - c, v)*dx
    for t in range(1, 5):
        solve(F == 0, u)
        u0.assign(u)
        adj_inc_timestep(t, t == 4)
    return u0

c = Constant(3.)
c = project(c, U, name="Control")
u = forward(c)

# check for one coord
Regform = Constant(0.01)*inner(c, c)*inner(c, c)*dx
J = PointwiseFunctional(u, [0, 0, 0, 0], Point(np.array([0.4, 0.4])), [1, 2, 3, 4], u_ind=[None], regularisation=Regform, verbose=False)
Jr = ReducedFunctional(J, Control(c))

Jr3 = Jr(c)
#G = compute_gradient(J, Control(c))
#print assemble(G)
#assert abs(Jr3 - 310.5) < 1e-4
assert Jr.taylor_test(project(Constant(5.), U)) > 1.9

