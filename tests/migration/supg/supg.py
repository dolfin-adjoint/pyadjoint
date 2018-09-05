# Copyright (C) 2007 Kristian B. Oelgaard
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
# Modified by Anders Logg, 2008
# Modified by Johan Hake, 2008
# Modified by Garth N. Wells, 2009
#
# This demo solves the time-dependent convection-diffusion equation by
# a SUPG stabilized method. The velocity field used
# in the simulation is the output from the Stokes (Taylor-Hood) demo.
# The sub domains for the different boundary conditions are computed
# by the demo program in src/demo/subdomains.
#

from fenics import *
from fenics_adjoint import *

from numpy.testing import assert_approx_equal


def boundary_value(n):
    if n < 10:
        return float(n)/10.0
    else:
        return 1.0

# Load mesh and subdomains
mesh = Mesh("mesh.xml.gz")
try:
    sub_domains = MeshFunction("sizet", mesh, "subdomains.xml.gz");
except:
    sub_domains = MeshFunction("size_t", mesh, "subdomains.xml.gz");
h = 2*Circumradius(mesh)

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function from file
velocity = Function(V);
File("velocity.xml.gz") >> velocity


def main(u0, f):

    # Parameters
    T = 1.0
    dt = 0.1
    t = dt
    c = 0.00005

    # Test and trial functions
    u, v = TrialFunction(Q), TestFunction(Q)
    u_new = Function(Q)

    # Mid-point solution
    u_mid = 0.5*(u0 + u)

    # Residual
    r = u-u0 + dt*(dot(velocity, grad(u_mid)) - c*div(grad(u_mid)) - f)

    # Galerkin variational problem
    F = v*(u-u0)*dx + dt*(v*dot(velocity, grad(u_mid))*dx + c*dot(grad(v), grad(u_mid))*dx)

    # Add SUPG stabilisation terms
    vnorm = sqrt(dot(velocity, velocity))
    F += (h/2.0*vnorm)*dot(velocity, grad(v))*r*dx

    # Create bilinear and linear forms
    a = lhs(F)
    L = rhs(F)

    # Set up boundary condition
    g = Expression("b", degree=1, b=boundary_value(0))
    bc = DirichletBC(Q, g, sub_domains, 1)

    # Assemble matrix
    A = assemble(a)
    bc.apply(A)

    # Create linear solver and factorize matrix
    solver = LUSolver()
    solver.set_operator(A)

    # Output file
    out_file = File("results/temperature.pvd")

    # Set intial condition
    u_new.assign(u0)

    # Time-stepping
    while t < T:
        # Assemble vector and apply boundary conditions
        b = assemble(L)
        bc.apply(b)

        # Solve the linear system (re-use the already factorized matrix A)
        solver.solve(u_new.vector(), b)

        # Copy solution from previous interval
        u0.assign(u_new)

        # Save the solution to file
        out_file << (u_new, t)

        # Move to next interval and adjust boundary condition
        t += dt
        g.b = boundary_value(int(t/dt))

    return u0

if __name__ == "__main__":
    u0 = Function(Q, name="Tracer")
    f  = Constant(0.0)
    u = main(u0, f=f)

    J = assemble(u*u*dx)
    param = Control(f)
    Jhat = ReducedFunctional(J, param)
    Jhat.optimize()
    assert_approx_equal(Jhat(f), J)

    dJdf = compute_gradient(J, param)
    h = Constant(1.0)
    dJdm = h._ad_dot(dJdf)

    minconv = taylor_test(ReducedFunctional(J, param), f, h, dJdm=dJdm)
    assert minconv > 1.9
