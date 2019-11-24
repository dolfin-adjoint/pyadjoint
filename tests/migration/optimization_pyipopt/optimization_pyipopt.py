""" Solves an optimisation problem with the Burgers equation as constraint, using
the pyipopt Python bindings to IPOPT"""

import sys

from fenics import *
from fenics_adjoint import *
import numpy

try:
    from pyadjoint import ipopt  # noqa
except ImportError:
    print("ipopt bindings unavailable, skipping test")
    sys.exit(0)

set_log_level(LogLevel.ERROR)

n = 10
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u_next, u, timestep):
    return (u_next - u)/timestep

def main(u, annotate=False):

    u_next = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u_next, u, timestep)*v
         + u_next*u_next.dx(0)*v + nu*u_next.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    while (t <= end):
        solve(F == 0, u_next, bc, annotate=annotate)
        u.assign(u_next, annotate=annotate)

        t += float(timestep)

def derivative_cb(j, dj, m):
    print("j = %f, max(dj) = %f, max(m) = %f." % (j, dj.vector().max(), m.vector().max()))

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    u = Function(V)

    # Run the model once to create the annotation
    u.assign(ic)
    main(u, annotate=True)

    J = assemble(u*u*dx)

    # Run the optimisation
    lb = interpolate(Constant(-1),  V)
    ub = interpolate(Constant(1.0e200), V)

    # Define the reduced funtional
    rf = ReducedFunctional(J, Control(ic))

    problem = MinimizationProblem(rf, bounds=(lb, ub))

    solver = IPOPTSolver(problem)
    m = solver.solve()

    Jm = rf(m)
    assert Jm < 1.0e-10
