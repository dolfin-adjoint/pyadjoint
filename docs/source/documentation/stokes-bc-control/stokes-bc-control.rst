..  #!/usr/bin/env python
  # -*- coding: utf-8 -*-
  
.. py:currentmodule:: dolfin_adjoint

Dirichlet BC control of the Stokes equations
============================================

.. sectionauthor:: Simon W. Funke <simon@simula.no>, André Massing <massing@simula.no>

This example demonstrates how to compute the sensitivity with respect to the Dirichlet
boundary conditions in pyadjoint.

Problem definition
******************

Consider the Stokes equations

.. math::
      -\nu \Delta u + \nabla p &= 0  \qquad \mathrm{in} \ \Omega \\
                        \mathrm{div }\  u &= 0  \qquad \mathrm{in} \ \Omega  \\

with Dirichlet boundary conditions

.. math::
          u &= g  \qquad \mathrm{on} \ \partial \Omega_{\textrm{cirlce}} \\
          u &= f  \qquad \mathrm{on} \ \partial \Omega_{\textrm{in}} \\
          u &= 0  \qquad \mathrm{on} \ \partial \Omega_{\textrm{walls}} \\
          p &= 0  \qquad \mathrm{on} \ \partial \Omega_{\textrm{out}} \\


where :math:`\Omega` is the domain of interest (visualised below),
:math:`u:\Omega \to \mathbb R^2` is the unknown velocity,
:math:`p:\Omega \to \mathbb R` is the unknown pressure, :math:`\nu`
is the viscosity, :math:`\alpha` is the regularisation parameter,
:math:`f` denotes the value for the Dirichlet inflow boundary
condition, and :math:`g` is the control variable that specifies the
Dirichlet boundary condition on the circle.

.. image:: stokes_bc_control_domain.png
    :scale: 35
    :align: center

The goal is to compute the sensitivity of the functional

.. math::
       \frac{1}{2}\int_{\Omega} \nabla u \cdot \nabla u~\textrm{d}x \\

Implementation
**************

First, the :py:mod:`fenics` and :py:mod:`fenics_adjoint` modules are imported:

::

  from fenics import *
  from fenics_adjoint import *
  
Next, we load the mesh. The mesh was generated with mshr; see make-mesh.py
in the same directory.

::

  mesh_xdmf = XDMFFile(mpi_comm_world(), "rectangle-less-circle.xdmf")
  mesh = Mesh()
  mesh_xdmf.read(mesh)
  
  
Then, we define the discrete function spaces. A Taylor-Hood
finite-element pair is a suitable choice for the Stokes equations.
The control function is the Dirichlet boundary value on the velocity
field and is hence be a function on the velocity space (note: FEniCS
cannot restrict functions to boundaries, hence the control is
defined over the entire domain).


::

  V_h = VectorElement("CG", mesh.ufl_cell(), 2)
  Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
  W = FunctionSpace(mesh, V_h * Q_h)
  V, Q = W.split()
  
  v, q = TestFunctions(W)
  x = TrialFunction(W)
  u, p = split(x)
  s = Function(W, name="State")
  V_collapse = V.collapse()
  g = Function(V_collapse, name="Control")
  
  
  # Set parameter values
  nu = Constant(1)     # Viscosity coefficient
  
  # Define boundary conditions
  u_inflow = Expression(("x[1]*(10-x[1])/25", "0"), degree=1)
  noslip = DirichletBC(W.sub(0), (0, 0),
                       "on_boundary && (x[1] >= 9.9 || x[1] < 0.1)")
  inflow = DirichletBC(W.sub(0), u_inflow, "on_boundary && x[0] <= 0.1")
  circle = DirichletBC(W.sub(0), g, "on_boundary && (pow((x[0]-10),2) + pow((x[1]-5),2)  < 3*3)")
  bcs = [inflow, noslip, circle]
  
  a = (nu*inner(grad(u), grad(v))*dx
       - inner(p, div(v))*dx
       - inner(q, div(u))*dx
       )
  f = Function(V_collapse)
  L = inner(f, v)*dx
  
  A, b = assemble_system(a, L, bcs)
  solve(A, s.vector(), b)
  
  u, p = split(s)
  alpha = Constant(10)
  
  J = assemble(1./2*inner(u, u)**2*dx)
  dJdm = compute_gradient(J, g)
  
  h = Function(V_collapse)
  h.vector()[:] = 1
  taylor_test(ReducedFunctional(J,g),g.copy(deepcopy = True),h,dJdm=dJdm.vector().inner(h.vector()))
