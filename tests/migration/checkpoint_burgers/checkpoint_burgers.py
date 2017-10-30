"""
Naive implementation of Burgers' equation, goes oscillatory later
"""

import sys

from dolfin import *
from dolfin_adjoint import *
from math import ceil

n = 100
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)


def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):
    timestep = Constant(1.0/n)
    t = 0.0
    end = 0.5
    if annotate:
        # TODO: Implement checkpointing.
        adj_checkpointing('multistage', int(ceil(end/float(timestep))), 5, 10, verbose=True)

    u_ = ic.copy(deepcopy=True, annotate=annotate)
    u = TrialFunction(V)
    v = TestFunction(V)
    nu = Constant(0.0001)

    F = (Dt(u, u_, timestep)*v
         + u_*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    (a, L) = system(F)

    bc = DirichletBC(V, 0.0, "on_boundary")

    u = Function(V)
    j = 0
    j += 0.5*AdjFloat(timestep)*assemble(u_*u_*dx)

    while (t <= end):
        solve(a == L, u, bc, annotate=annotate)

        u_.assign(u, annotate=annotate)

        t += float(timestep)

        if t>end:
            quad_weight = 0.5
        else:
            quad_weight = 1.0
        j += quad_weight*AdjFloat(timestep)*assemble(u_*u_*dx)

    return j, u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    j, forward = main(ic, annotate=True)

    J = j
    m = Control(ic)
    dJdm = compute_gradient(J, m)
    h = Function(V)
    h.vector()[:] = 1
    dJdm = h._ad_dot(dJdm)

    def Jhat(ic):
        j, forward = main(ic, annotate=False)
        return j

    minconv = taylor_test(Jhat, ic, h, dJdm)
    assert minconv > 1.9
