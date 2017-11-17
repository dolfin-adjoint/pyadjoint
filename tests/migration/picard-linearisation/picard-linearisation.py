r"""
See

src/snes/examples/tutorials/ex15.c

in the PETSc source.

Solve the nonlinear diffusion problem

-div( \gamma \nabla u ) = f

where

f = 0.1

and

\gamma = (\epsilon^2 + 0.5 * |\nabla u|^2)^{(p-2)/2}

where

\epsilon = 1.0e-5.
"""

from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "CG", 1)
epsilon = Constant(1.0e-5)
p = Constant(3.0)

parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
picard = True
bc = DirichletBC(V, 0.0, "on_boundary")

def problem(u_trial, u_guess, v, f, p):
    F = inner(grad(v), gamma(u_guess, p) * grad(u_trial))*dx - inner(f*f, v)*dx

    return F

def gamma(u, p):
    return (epsilon**2 + 0.5 * inner(grad(u), grad(u)))**((p-2)/2)

def main(f):
    u = Function(V, name="Solution")
    trial = TrialFunction(V)
    test = TestFunction(V)

    nF = problem(u, u, TestFunction(V), f, p=p) # suitable for Newton iteration
    F = problem(trial, u, test, f, p=p)     # suitable for Picard iteration
    a = lhs(F)

    if picard:
        J = a
        damping = 0.5
    else:
        J = derivative(nF, u)
        damping = 1.0

    solve(nF == 0, u, J=J, bcs=bc, solver_parameters={"newton_solver": {"maximum_iterations": 200, "relaxation_parameter": damping}})

    return u

if __name__ == "__main__":
    f = interpolate(Expression("sin(x[0])*cos(x[1])", degree=1), V)
    f = Function(V, f.vector(), name="SourceTerm")
    u = main(f)

    if False:
        # TODO: Not implemented
        assert replay_dolfin(tol=0.0, stop=True)
    J = assemble(inner(u, u)*dx)
    m = Control(f)
    dJdm = compute_gradient(J, m)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJdm = h._ad_dot(dJdm)

    def Jhat(f):
        u = main(f)
        return assemble(inner(u, u)*dx)

    minconv = taylor_test(Jhat, f, h, dJdm)
    assert minconv > 1.8
