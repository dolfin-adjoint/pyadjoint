import io
import sys

from dolfin import *
from dolfin_adjoint import *
import libadjoint

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "R", 0)
f = Constant(1.0)
g = Constant(2.0)

# Solve the 'PDE' with solution u(t) = t
def run_forward(annotate=True):
    u = TrialFunction(V)
    v = TestFunction(V)

    u_0 = Function(V, name="Value")
    u_obs = Function(V, name="ObsValue")

    dt = 0.5
    T =  1.0

    F = ( (u - u_0)/dt*v - f*v)*dx
    a, L = lhs(F), rhs(F)

    F_obs = ( (u - u_obs)/dt*v - g*v)*dx
    a_obs, L_obs = lhs(F_obs), rhs(F_obs)

    t = float(dt)

    #print "u_0.vector().array(): ", u_0.vector().array()
    adjointer.time.start(0)
    while t <= T:

        solve(a == L, u_0, annotate=annotate)
        solve(a_obs == L_obs, u_obs, annotate=annotate)

        adj_inc_timestep(time=t, finished=t+dt>T)
        t += float(dt)

    return (u_0, u_obs)

if __name__ == "__main__":

    (u, u_obs) = run_forward()

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    u00 = libadjoint.Variable("Value", 0, 0)
    u01 = libadjoint.Variable("Value", 0, 1)
    u10 = libadjoint.Variable("Value", 1, 0)

    # Integral over all time
    J = Functional(inner(u,u)*dx*dt[0:1])
    assert J.dependencies(adjointer, 0) == [u00]
    deps=J.dependencies(adjointer, 1)
    assert deps[0] in [u01, u10]
    assert deps[1] in [u01, u10]
    assert deps[0] != deps[1]

    # Integral over a certain time window
    J = Functional(inner(u,u)*dx*dt[0.5:1.0])
    assert J.dependencies(adjointer, 0) == []
    deps=J.dependencies(adjointer, 1)
    assert deps[0] in [u01, u10]
    assert deps[1] in [u01, u10]
    assert deps[0] != deps[1]

    # Pointwise evaluation (in the middle of a timestep)
    J = Functional(inner(u,u)*dx*dt[0.25])
    # assert J.dependencies(adjointer, 0) == [u00, u01]
    deps=J.dependencies(adjointer, 0)
    assert deps[0] in [u00, u01]
    assert deps[1] in [u00, u01]
    assert deps[0] != deps[1]
    assert J.dependencies(adjointer, 1) == []

    # Pointwise evaluation (at a timelevel)
    J = Functional(inner(u,u)*dx*dt[0.5])
    assert J.dependencies(adjointer, 0) == []
    assert J.dependencies(adjointer, 1) == [u01]

    # Pointwise evaluation at final time
    J = Functional(inner(u,u)*dx*dt[1.0])
    assert J.dependencies(adjointer, 0) == []
    assert J.dependencies(adjointer, 1) == [u10]

    # Pointwise evaluation at final time, symbolic
    J = Functional(inner(u,u)*dx*dt[FINISH_TIME])
    assert J.dependencies(adjointer, 0) == []
    assert J.dependencies(adjointer, 1) == [u10]

    # Pointwise evaluation at start time
    J = Functional(inner(u,u)*dx*dt[0.0])
    assert J.dependencies(adjointer, 0) == [u00]
    assert J.dependencies(adjointer, 1) == []

    # Pointwise evaluation at start time, symbolic
    J = Functional(inner(u,u)*dx*dt[START_TIME])
    assert J.dependencies(adjointer, 0) == [u00]
    assert J.dependencies(adjointer, 1) == []

    # Integral over all time
    J = Functional(inner(u,u)*dx*dt[0:1])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.375

    # Integral over all time, no indices
    J = Functional(inner(u,u)*dx*dt)
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.375

    # Integral over the first time interval
    J = Functional(inner(u,u)*dx*dt[0:0.5])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.0625

    # Integral over the finishing interval.
    J = Functional(inner(u,u)*dx*dt[0.75:])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.15625

    # Integral over the first time interval with unspecified start
    J = Functional(inner(u,u)*dx*dt[:0.5])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.0625

    # Integral over something that's not aligned with the timesteps
    J = Functional(inner(u,u)*dx*dt[0.25:0.75])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.1875

    # Pointwise evaluation in time
    J = Functional(inner(u,u)*dx*dt[0.25])
    assert adjointer.evaluate_functional(J, 0) == 0.25**2
    assert adjointer.evaluate_functional(J, 1) == 0.0

    # Pointwise evaluation in time
    J = Functional(inner(u,u)*dx*dt[0.5])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.5**2

    # Pointwise evaluation at final time
    J = Functional(inner(u,u)*dx*dt[1.0])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 1.0

    # Pointwise evaluation at final time, symbolic
    J = Functional(inner(u,u)*dx*dt[FINISH_TIME])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 1.0

    # Pointwise evaluation at start time
    J = Functional(inner(u,u)*dx*dt[0.0])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.0

    # Pointwise evaluation at final time, symbolic
    J = Functional(0 + inner(u,u)*dx*dt[START_TIME])
    assert adjointer.evaluate_functional(J, 0) == 0.0
    assert adjointer.evaluate_functional(J, 1) == 0.0

    o00 = libadjoint.Variable("ObsValue", 0, 0)
    o01 = libadjoint.Variable("ObsValue", 0, 1)
    o10 = libadjoint.Variable("ObsValue", 1, 0)

    # Investigate its behaviour with other variables also
    J = Functional(inner(u - u_obs, u - u_obs)*dx*dt)
    deps = J.dependencies(adjointer, 0)
    assert len(deps) == 2
    assert deps[0] in [u00, o00]
    assert deps[1] in [u00, o00]
    assert deps[0] != deps[1]

    deps = J.dependencies(adjointer, 1)
    assert len(deps) == 4
    assert deps[0] in [u01, u10, o01, o10]
    assert deps[1] in [u01, u10, o01, o10]
    assert deps[2] in [u01, u10, o01, o10]
    assert deps[3] in [u01, u10, o01, o10]
