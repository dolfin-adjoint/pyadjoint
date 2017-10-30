"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

from __future__ import print_function
import sys

from dolfin import *
from dolfin_adjoint import *

from numpy import array

dolfin.parameters["adjoint"]["record_all"] = True
dolfin.parameters["adjoint"]["fussy_replay"] = False

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic.copy(deepcopy=True, name="Velocity")
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Function(V)
    nu.vector()[:] = 0.0001

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    while (t <= end):
        nu.vector()[:] += array([0.0001] * V.dim()) # <--------- change nu here by hand
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

    return u_

if __name__ == "__main__":
    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    main(ic, annotate=True)

    success = replay_dolfin(forget=False) # <------ should catch it here
    assert not success
