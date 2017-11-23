from dolfin import *
from dolfin_adjoint import *
import numpy as np
np.random.seed(seed=21) 


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

mJ = PointwiseFunctional(u, [[0, 0, 0, 0], [1, 1, 1, 1]], [Point(np.array([0.2, 0.2])), Point(np.array([0.8, 0.8]))], [1, 2, 3, 4], u_ind=[None, None])
mJr = ReducedFunctional(mJ, Control(c))

mJr3 = mJr(Constant(3))

assert mJr.taylor_test(Constant(5), seed=1e2) > 1.9
assert abs(mJr3 - 484.0) < 1e-11, abs(mJr3 - 484.0)
info_green("Test passed")
