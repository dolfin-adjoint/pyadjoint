import sys

import kelvin_new as kelvin
import sw_lib

from fenics import *
from fenics_adjoint import *

from numpy.random import rand, seed

mesh = UnitSquareMesh(6, 6)
W=sw_lib.p1dgp2(mesh)

state=Function(W)

state.interpolate(kelvin.InitialConditions(degree=1))
m = Control(state)

kelvin.params["basename"] = "p1dgp2"
kelvin.params["dt"] = 2
kelvin.params["finish_time"] = kelvin.params["dt"]*2
kelvin.params["dump_period"] = 1

M, G=sw_lib.construct_shallow_water(W, kelvin.params)

state = sw_lib.timeloop_theta(M, G, state, kelvin.params)

if False:
    # TODO: Not implemented.
    replay_dolfin()

J = assemble(dot(state, state)*dx)
ic = Function(W)
ic.interpolate(kelvin.InitialConditions(degree=1))
def compute_J(ic):
    state = sw_lib.timeloop_theta(M, G, ic.copy(deepcopy=True), kelvin.params, annotate=False)
    return assemble(dot(state, state)*dx)

h = Function(W)
h.vector()[:] = rand(W.dim())*0.01
dJdm = compute_gradient(J, m)
dJdm = h._ad_dot(dJdm)

minconv = taylor_test(compute_J, ic, h, dJdm)
assert minconv > 1.9
