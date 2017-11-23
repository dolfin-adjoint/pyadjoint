from dolfin import *
from dolfin_adjoint import *
import numpy as np

# Define mesh
mesh = UnitSquareMesh(10, 10)
n    = FacetNormal(mesh)
U    = FunctionSpace(mesh, "CG", 1)

adj_start_timestep()

def forward(c):
    u  = Function(U)
    u0 = Function(U)
    v  = TestFunction(U)
    F  = inner(u - u0 - c, v)*dx
    for t in range(1, 5):
        solve(F == 0, u)
        u0.assign(u)
        adj_inc_timestep(t, t == 4)
    return u0

c = Constant(3.)
u = forward(c)

J = PointwiseFunctional(u, [0, 0, 0, 0], Point(np.array([0.4, 0.4])), [1, 2, 3, 4], u_ind=[None])
Jr = ReducedFunctional(J, Control(c))

Jr3 = Jr(Constant(3))

assert Jr.taylor_test(Constant(5)) > 1.9
assert abs(Jr3 - 270.0) < 1e-12
info_green("Test passed")
