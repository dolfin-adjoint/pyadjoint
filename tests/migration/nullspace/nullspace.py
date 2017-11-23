"""
This demo program illustrates how to solve Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with pure Neumann boundary conditions:

    du/dn(x, y) = -sin(5*x)

and source f given by

    f(x, y) = 10*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

Since only Neumann conditions are applied, u is only determined up to
a constant by the above equations. An addition constraint is thus
required, for instance

  \int u = 0

This is accomplished in this demo by using a Krylov iterative solver
that removes the component in the null space from the solution vector.
"""

# Copyright (C) 2012 Garth N. Rognes
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2012-10-31
# Last changed: 2012-11-12

from dolfin import *
from dolfin_adjoint import *

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    info("DOLFIN has not been configured with TPETSc. Exiting.")
    exit()

if dolfin.__version__ < "1.2.0+":
    info("Need dolfin > 1.2.0")
    exit()

parameters["linear_algebra_backend"] = "PETSc"

# Create mesh and define function space
mesh = UnitSquareMesh(24, 24)
V = FunctionSpace(mesh, "CG", 1)
null_space = Vector(Function(V).vector())
V.dofmap().set(null_space, 1.0)
null_space[:] = null_space/null_space.norm("l2")

def main(f, g):
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx + g*v*ds

    # Assemble system
    A = assemble(a)
    b = assemble(L)

    #bc = DirichletBC(V, 0.0, "on_boundary"); bc.apply(A); bc.apply(b)

    # Solution Function
    u = Function(V)

    # Create Krylov solver
    solver = KrylovSolver(A, "gmres")

    # Create null space basis and attach to Krylov solver
    null_vec = Vector(u.vector())
    V.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0/null_vec.norm("l2")
    null_space = VectorSpaceBasis([null_vec])
    solver.set_nullspace(null_space)

    # In this case, the system is symmetric, so the transpose nullspace is the same
    solver.set_transpose_nullspace(null_space);

    # When solving singular systems, you have to ensure that the RHS b is in the range
    # of the singular matrix. Since the range is the orthogonal complement of the
    # transpose nullspace, you have to call the orthogonalize method of the transpose
    # nullspace object on the RHS:
    null_space.orthogonalize(b);

    solver.parameters["relative_tolerance"] = 1.0e-200
    solver.parameters["absolute_tolerance"] = 1.0e-14
    solver.parameters["maximum_iterations"] = 20000

    # Solve
    solver.solve(u.vector(), b)

    return u

if __name__ == "__main__":
    g = interpolate(Expression("-sin(5*x[0])", degree=1), V, name="SourceG")
    f = interpolate(Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1), V, name="SourceF")
    u = main(f, g)

    parameters["adjoint"]["stop_annotating"] = True

    assert replay_dolfin(tol=0.0, stop=True)

    frm = lambda u: inner(u, u)*dx

    Jm = assemble(frm(u))
    J = Functional(frm(u))
    m = Control(f)
    dJdm = compute_gradient(J, m, forget=False)

    def Jhat(f):
        u = main(f, g)
        return assemble(frm(u))

    minconv = taylor_test(Jhat, m, Jm, dJdm, perturbation_direction=interpolate(Constant(1.0), V))
    assert minconv > 1.8
