..  #!/usr/bin/env python
  # -*- coding: utf-8 -*-
  
.. py:currentmodule:: dolfin_adjoint

Drag minimization over an obstacle in Stokes-flow
=================================================

.. sectionauthor:: Jørgen S. Dokken <dokken@simula.no>

This demo solves the famous shape optimization problem
for minimizing drag over an obstacle subject to stokes
flow. This problem was initially analyzed by :cite:`pironneau1974optimum`
where the optimal geometry was found to be a rugby shaped ball with
a 90 degree front and back wedge.

Problem definition
******************

This problem is to minimize

.. math::
      \min_{\sigma,u} \int_{\Omega(\sigma, s)} \sum_{i,j=1}^2 \left(
      \frac{\partial u_i}{\partial x_j}\right)






::

  from dolfin import *
  
  from create_mesh import inflow, outflow walls, obstacle
  
  mesh = Mesh()
  with XDMFFile("mesh.xdmf") as infile:
      infile.read(mesh)
      mvc = MeshValueCollection("size_t", mesh, 1)
  with XDMFFile("mf.xdmf") as infile:
      infile.read(mvc, "name_to_read")
      mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)
          
  
.. bibliography:: /documentation/stokes-shape-opt/stokes-shape-opt.bib
   :cited:
   :labelprefix: 1E-
