import io
import sys

from dolfin import *
from dolfin_adjoint import *
import libadjoint

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "R", 0)
f = Constant(1.0)

# Solve the 'PDE' du/dt = 1
# with initial condition u(0) = c
# this gives the solution u(t) = c + t.
def run_forward(annotate=True):
    u = TrialFunction(V)
    v = TestFunction(V)

    u_0 = Function(V, name="Value")
    u_0.vector()[0] = 1.0

    dt = 0.5
    T =  1.0

    F = ( (u - u_0)/dt*v - f*v)*dx
    a, L = lhs(F), rhs(F)

    t = float(dt)

    #print "u_0.vector().array(): ", u_0.vector().array()
    adjointer.time.start(0)
    while t <= T:

        solve(a == L, u_0, annotate=annotate)
        #print "u_0.vector().array(): ", u_0.vector().array()

        adj_inc_timestep(time=t, finished=t+dt>T)
        t += float(dt)

    return u_0

if __name__ == "__main__":

    u = run_forward()

    # Integral over all time
    J = Functional(inner(u,u)*dx*dt[0:1])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 3
    assert (dJdic.vector().array()[0] - 3.0) < 1e-15

    # Integral over a certain time window
    J = Functional(inner(u,u)*dx*dt[0.5:1.0])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 7/4
    assert (dJdic.vector().array()[0] - 7.0/4.0) < 1e-15

    # Pointwise evaluation (in the middle of a timestep)
    J = Functional(inner(u,u)*dx*dt[0.25])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 5/2
    assert (dJdic.vector().array()[0] - 2.5) < 1e-15

    # Pointwise evaluation (at a timelevel)
    J = Functional(inner(u,u)*dx*dt[0.5])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 3
    assert (dJdic.vector().array()[0] - 3.0) < 1e-15

    # Pointwise evaluation (at the end of time)
    J = Functional(inner(u,u)*dx*dt[1.0])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 4
    assert (dJdic.vector().array()[0] - 4.0) < 1e-15

    # Pointwise evaluation (at the end of time, symbolically)
    J = Functional(inner(u,u)*dx*dt[FINISH_TIME])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 4
    assert (dJdic.vector().array()[0] - 4.0) < 1e-15

    # Pointwise evaluation (at the start of time)
    J = Functional(inner(u,u)*dx*dt[0.0])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 2
    assert (dJdic.vector().array()[0] - 2.0) < 1e-15

    # Pointwise evaluation (at the start of time, symbolically)
    J = Functional(inner(u,u)*dx*dt[START_TIME])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 2
    assert (dJdic.vector().array()[0] - 2.0) < 1e-15

    # Let's do a sum
    J = Functional(inner(u,u)*dx*dt[0.5] + inner(u,u)*dx*dt[0.5:1.0])
    dJdic = compute_gradient(J, Control(u), forget=False)
    # Work out the solution by hand -- it's 3 + 7/4
    assert (dJdic.vector().array()[0] - 3.0 - 7.0/4.0) < 1e-15

    # Integral over all time, forgetting this time
    J = Functional(inner(u,u)*dx*dt[0:1])
    dJdic = compute_gradient(J, Control(u), forget=True)
    # Work out the solution by hand -- it's 3
    assert (dJdic.vector().array()[0] - 3.0) < 1e-15
