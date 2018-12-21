#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. py:currentmodule:: dolfin_adjoint
#
# Drag minimization over an obstacle in Stokes-flow
# =================================================
#
# .. sectionauthor:: Jørgen S. Dokken <dokken@simula.no>
#
# This demo solves the famous shape optimization problem
# for minimizing drag over an obstacle subject to Stokes
# flow. This problem was initially analyzed by :cite:`pironneau1974optimum`
# where the optimal geometry was found to be a rugby shaped ball with
# a 90 degree front and back wedge.
#
# Problem definition
# ******************
#
# This problem is to find the shape of the obstacle :math:`\Gamma`, which minimizes the dissipated power in the fluid
#
# .. math::
#       \min_{\sigma,u,s} \int_{\Omega(s)} \sum_{i,j=1}^2 \left(
#       \frac{\partial u_i}{\partial x_j}\right)^2~\mathrm{d} x
#        +\left(\vert\Omega_0\vert - \int_{\Omega(s)}1\mathrm{d} x \right)^2,
#
# where :math:`\vert\Omega_0\vert` is the initial volume.
# In :math:`J`, u is a velocity field subject to the Stokes equations:
#
# .. math::
#       -\Delta u + \nabla p &= 0 \qquad \mathrm{in} \ \Omega(s), \\
#       \mathrm{div}(u) &= 0 \qquad \mathrm{in} \ \Omega(s), \\
#       u &= 0 \qquad \mathrm{on} \ \Gamma(s)\cup\Lambda_1,\\
#       u &= g \qquad \mathrm{on} \ \Lambda_2, \\
#       \frac{\partial u }{\partial n} + pn &= 0 \qquad \mathrm{on} \ \Lambda_3,
#    :label: state
#
# where :math:`\Lambda_1` are the walls, :math:`\Lambda_2` the inlet
# and :math:`\Lambda_3` the outlet of the channel.
#
# We define the change of the fluid domain from its unperturbed state
# :math:`\Omega_0`, as :math:`\Omega(s)=\{x+s(\sigma)\vert x\in \Omega_0 \}`,
# , and :math:`s` is a weighted
# :math:`H^1(\Omega)`-smoothing solving.
#
# .. math::
#       -\alpha\Delta s + s &= 0 \qquad \text{in} \ \Omega_0,\\
#       s&=0 \qquad \text{on} \ \Lambda_1\cup\Lambda_2\cup\Lambda_3,\\
#       \frac{\partial s}{\partial n} &= \sigma \qquad \text{on} \ \Gamma.\\
#    :label: deformation 
#    
#
#
# Implementation
# **************
#
# First, the :py:mod:`dolfin`and :py:mod:`dolfin_adjoint` modules are imported:

from dolfin import *
from dolfin_adjoint import *


# Next, we load the facet marker values used in the mesh from the
# mesh-generator file.

from create_mesh import inflow, outflow, walls, obstacle

# The initial (unperturbed) mesh and corresponding facet function from their respective
# xdmf-files.

mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("mf.xdmf") as infile:
    infile.read(mvc, "name_to_read")
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# We compute the initial volume of the computational domain

Omega0 = assemble(Constant(1)*dx(domain=mesh))

# We create a Boundary-mesh and function space for our control :math:`\sigma`

b_mesh = BoundaryMesh(mesh, "exterior")
S_b = VectorFunctionSpace(b_mesh, "CG", 1)
sigma = Function(S_b, name="Design")

# We create a corresponding function space on :math:`\Omega`, and
# transfer the corresponding boundary values to the function
# :math:`\sigma_V`. This call is needed to be able to represent
# :math:`\sigma` in the variational form of :math:`s`.

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Mesh perturbation field")
sigma_V = transfer_from_boundary(sigma, mesh)
sigma_V.rename("Volume extension of sigma", "")

# We can now transfer our mesh according to :eq:`deformation`.

u, v = TrialFunction(S), TestFunction(S)
alpha = 0.1
fixed = Constant((0,0))
a = alpha*inner(grad(u),grad(v))*dx + inner(u,v)*dx
dGamma = Measure("ds", domain=mesh,
                    subdomain_data=mf, subdomain_id=obstacle)
bc_inlet = DirichletBC(S, fixed, mf, inflow)
bc_outlet = DirichletBC(S, fixed, mf, outflow)
bc_walls = DirichletBC(S, fixed, mf, walls)
bc_deform = [bc_inlet, bc_outlet, bc_walls]
l = inner(sigma_V, v)*dGamma
solve(a==l, s, bcs=bc_deform)
ALE.move(mesh, s)

# The next step is to set up :eq:`state`. We start by defining the
# stable Taylor-Hood finite element space.

V2 = VectorElement("CG", mesh.ufl_cell(), 2)
S1 = FiniteElement("CG", mesh.ufl_cell(), 1)
VQ = FunctionSpace(mesh, V2*S1)

# Then, we define the test and trial functions, as well as the variational form

(u, p) = TrialFunctions(VQ)
(v, q) = TestFunctions(VQ)
a = inner(grad(u), grad(v))*dx - div(u)*q*dx - div(v)*p*dx
l = inner(Constant((0,0)), v)*dx

# The Dirichlet boundary conditions on :math:`\Gamma` is defined as follows

(x,y) = SpatialCoordinate(mesh)
g = Expression(("sin(pi*x[1])","0"),degree=2)
noslip = Constant((0,0))
bc_inlet = DirichletBC(VQ.sub(0), g, mf, inflow)
bc_obstacle = DirichletBC(VQ.sub(0),noslip , mf, obstacle)
bc_walls = DirichletBC(VQ.sub(0), noslip, mf, walls)
bcs = [bc_inlet, bc_obstacle, bc_walls]

w = Function(VQ, name="Mixed State Solution")
solve(a==l, w, bcs=bcs)
u, p = w.split()

J = assemble(inner(grad(u), grad(u))*dx)
J += (Omega0 - assemble(Constant(1)*dx(domain=mesh)))**2

Jhat = ReducedFunctional(J, Control(sigma))

# The computational tape can be visualized with the following commands

tape = get_working_tape()
tape.visualise()

# We perform a Taylor-test to verify the shape gradient

perturbation = interpolate(Expression(("A*sin(x[0])", "A*x[1]"),
                                      A=2,degree=2), S_b)
s_0 = Function(S_b) # Initial point in taylor-test
results = taylor_to_dict(Jhat, s_0, perturbation)

# We check that we obtain the expected convergence rates for the
# Finite Difference, with gradient information and with Hessian information
print("Finite Difference residuals")
print(" ".join("{:.2e}".format(res) for res in results["FD"]["Residual"]))
print("Residuals with gradient info")
print(" ".join("{:.2e}".format(res) for res in results["dJdm"]["Residual"]))
print("Residuals with hessian info")
print(" ".join("{:.2e}".format(res) for res in results["Hm"]["Residual"]))
print("*"*15)
print("Finite difference convergence rates")
print(" ".join("{:2.3f}".format(res) for res in results["FD"]["Rate"]))
print("Convergence rate with gradient info")
print(" ".join("{:2.3f}".format(res) for res in results["dJdm"]["Rate"]))
print("Convergence rate with hessian info")
print(" ".join("{:2.3f}".format(res) for res in results["Hm"]["Rate"]))
assert(min(results["FD"]["Rate"])>0.9)
assert(min(results["dJdm"]["Rate"])>1.95)
assert(min(results["Hm"]["Rate"])>2.95)


# We visualize the maximum displacement in the taylor test
unperturbed, _ = plot(mesh, color="k", label="unperturbed", linewidth=0.5)
perturbation.vector()[:]*=0.01 # Taylor test scales internally
Jhat(perturbation)
perturbed, _ = plot(mesh, color="r", label="perturbed",linewidth=0.5)
import matplotlib.pyplot as plt
plt.axis("off")
plt.legend(handles=[unperturbed, perturbed])
plt.savefig("mesh.png",dpi=250,bbox_inches="tight",
            pad_inches=0)

# .. bibliography:: /documentation/stokes-shape-opt/stokes-shape-opt.bib
#    :cited:
#    :labelprefix: 1E-
