from dolfin import *
from dolfin_adjoint import *

import numpy as np

# Define mesh
mesh = UnitSquareMesh(10, 10)
n    = FacetNormal(mesh)
U    = VectorFunctionSpace(mesh, "CG", 1)

adj_start_timestep()

def forward(c):
    u  = Function(U, name="u")
    u0 = Function(U, name="u0")
    v  = TestFunction(U)
    F  = inner(u - u0 - c*Constant((1., 2)), v)*dx
    for t in range(1, 5):
        solve(F == 0, u)
        u0.assign(u)
        adj_inc_timestep(t, t == 4)
    return u0

c = Constant(3.)
u = forward(c)

J0 = PointwiseFunctional(u, [0, 0, 0, 0], Point(np.array([0.4, 0.4])), [1, 2, 3, 4], u_ind=[0])
Jr0 = ReducedFunctional(J0, Control(c))
J1 = PointwiseFunctional(u, [0, 0, 0, 0], Point(np.array([0.4, 0.4])), [1, 2, 3, 4], u_ind=[1], alpha=0.5)
Jr1 = ReducedFunctional(J1, Control(c))

Jr3 = Jr0(Constant(3.))
Jr4 = Jr1(Constant(2.))

assert abs(Jr3 - 270.0) < 1e-12
assert abs(Jr4 - 240.0) < 1e-11
assert Jr0.taylor_test(Constant(5.)) > 1.9

