""" Steady state advection-diffusion equation,
discontinuous formulation using full upwinding.

Implemented in python from cpp demo by Johan Hake.
"""

# Copyright (C) 2008 Johan Hake
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

from fenics import *
from fenics_adjoint import *
import sys

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

# Load mesh
mesh = Mesh("mesh.xml.gz")

# Defining the function spaces
V_dg = FunctionSpace(mesh, "DG", 1)
V_cg = FunctionSpace(mesh, "CG", 1)
V_u  = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function
u = Function(V_u, "velocity.xml.gz")

def main(kappa):
    # Test and trial functions
    v   = TestFunction(V_dg)
    phi = TrialFunction(V_dg)

    # Diffusivity

    # Source term
    f = Constant(0.0)

    # Penalty term
    alpha = Constant(5.0)

    # Mesh-related functions
    n = FacetNormal(mesh)
    h = 2*Circumradius(mesh)
    h_avg = (h('+') + h('-'))/2

    # ( dot(v, n) + |dot(v, n)| )/2.0
    un = (dot(u, n) + abs(dot(u, n)))/2.0

    # Bilinear form
    a_int = dot(grad(v), kappa*grad(phi) - u*phi)*dx

    a_fac = kappa('+')*(alpha('+')/h('+'))*dot(jump(v, n), jump(phi, n))*dS \
          - kappa('+')*dot(avg(grad(v)), jump(phi, n))*dS \
          - kappa('+')*dot(jump(v, n), avg(grad(phi)))*dS

    a_vel = dot(jump(v), un('+')*phi('+') - un('-')*phi('-') )*dS  + dot(v, un*phi)*ds

    a = a_int + a_fac + a_vel

    # Linear form
    L = v*f*dx

    # Set up boundary condition (apply strong BCs)
    g = Expression("sin(pi*5.0*x[1])", degree=1)
    bc = DirichletBC(V_dg, g, DirichletBoundary(), "geometric")

    # Solution function
    phi_h = Function(V_dg)

    # Assemble and apply boundary conditions
    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)

    # Solve system
    solve(A, phi_h.vector(), b)

    return phi_h

if __name__ == "__main__":
    kappa = Constant(0.0, name="Kappa")

    phi = main(kappa)

    if False:
        # TODO: Not implemented.
        success = replay_dolfin()
        if not success:
            sys.exit(1)

    J = assemble(phi*phi*dx)
    dJdkappa = compute_gradient(J, Control(kappa))
    h = Constant(0.001)
    dJdkappa = h._ad_dot(dJdkappa)

    def J(kappa):
        phi = main(kappa)
        return assemble(phi*phi*dx)

    minconv = taylor_test(J, kappa, h, dJdkappa)

    if minconv < 1.9:
        sys.exit(1)
