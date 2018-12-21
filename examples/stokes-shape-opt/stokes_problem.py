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


# We create a Boundary-mesh and function space for our control :math:`\sigma`

b_mesh = BoundaryMesh(mesh, "exterior")
V_b = VectorFunctionSpace(b_mesh, "CG", 1)
sigma = Function(V_b, name="Design")

# We create a corresponding function space on :math:`\Omega`, and
# transfer the corresponding boundary values to the function
# :math:`\sigma_V`. This call is needed to be able to represent
# :math:`\sigma` in the variational form of :math:`s`.

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Mesh perturbation field")
sigma_V = transfer_from_boundary(sigma, mesh)
sigma_V.rename("Volume extension of sigma", "")

# We can now transfer our mesh according to :eq:`deformation`.
#
#





# .. bibliography:: /documentation/stokes-shape-opt/stokes-shape-opt.bib
#    :cited:
#    :labelprefix: 1E-
