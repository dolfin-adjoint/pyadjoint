from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "CG", 1)

v = TestFunction(V)
u_new = Function(V)
u_old = Function(V)

a = inner(u_new - u_old - Constant(1),  v)*dx

adjointer.time.start(0)
for t in range(5):
    solve(a == 0, u_new)
    adj_inc_timestep(time=t+1, finished=(t+1==5))
    u_old.assign(u_new)

adj_html("forward.html", "forward")

# Get the tape value of the last timestep
var = DolfinAdjointVariable(u_new)
assert var.timestep == 4
assert var.iteration == 0
assert max(abs(var.tape_value().vector().array() - 5)) < 1e-12

assert var.known_timesteps() == list(range(5))

# Alternatively we can use negative indices to get the latest Variable
var = DolfinAdjointVariable(u_new, timestep=-2, iteration=-1)
assert var.timestep == 4
assert var.iteration == 0
assert max(abs(var.tape_value().vector().array() - 5)) < 1e-12

# Another way is to explicitly set timestep and iteration to retrieve
for t in range(5):
    var = DolfinAdjointVariable(u_new, timestep=t, iteration=-1)
    expected_value = t + 1
    assert max(abs(var.tape_value().vector().array() - expected_value)) < 1e-12
