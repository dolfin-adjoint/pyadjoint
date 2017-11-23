from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "R", 0)

test = TestFunction(V)
trial = TrialFunction(V)

def main(m):
    u = interpolate(Constant(0.1), V, name="Solution")

    F = inner(u*u, test)*dx - inner(m, test)*dx
    solve(F == 0, u)
    F = inner(sin(u)*u*u*trial, test)*dx - inner(u**4, test)*dx
    solve(lhs(F) == rhs(F), u)

    return u

if __name__ == "__main__":
    m = interpolate(Constant(2.13), V, name="Parameter1")
    u = main(m)

    J = assemble(inner(u, u)**3*dx + inner(m, m)*dx)

    controls = [Control(m)]

    dJdm = compute_gradient(J, controls)
    HJm  = Hessian(J, controls)

    def Jhat(m):
        m = m[0] # the control is a list of length one, so Jhat will have to
        # except a list as well
        u = main(m)
        return assemble(inner(u, u)**3*dx + inner(m, m)*dx)

    direction = [interpolate(Constant(0.1), V)]
    minconv = taylor_test_multiple(Jhat, controls, Jm, dJdm, HJm=HJm,
                          perturbation_direction=direction)
    assert minconv > 2.9
