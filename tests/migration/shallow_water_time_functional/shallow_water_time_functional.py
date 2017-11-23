import divett
import sw_lib
from fenics import *
from fenics_adjoint import *

from numpy.random import rand

W=sw_lib.p1dgp2(divett.mesh)

state=Function(W)

state.interpolate(divett.InitialConditions(degree=1))
m = Control(state)

divett.params["basename"]="p1dgp2"
divett.params["finish_time"]=2*pi/(sqrt(divett.params["g"]*divett.params["depth"])*pi/3000)
divett.params["dt"]=divett.params["finish_time"]/5
divett.params["period"]=60*60*1.24
divett.params["dump_period"]=1

M, G, rhs_contr, ufl,ufr=sw_lib.construct_shallow_water(W, divett.ds, divett.params)

j, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, ufr, state, divett.params)

if False:
    # TODO: Not implemented.
    replay_dolfin()

(u,p) = split(state)
J = j + assemble(3.14*dot(state, state)*dx)

def compute_J(ic):
    j, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, ufr, ic.copy(deepcopy=True), divett.params, annotate=False)
    return j + assemble(3.14*dot(state, state)*dx)

ic = Function(W)
ic.interpolate(divett.InitialConditions(degree=1))

h = Function(W)
h.vector()[:] = rand(W.dim())
dJdm = compute_gradient(J, m)
dJdm = h._ad_dot(dJdm)

minconv = taylor_test(compute_J, ic, h, dJdm)
assert minconv > 1.9
