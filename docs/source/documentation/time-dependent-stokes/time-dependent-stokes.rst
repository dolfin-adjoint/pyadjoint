.. py:currentmodule:: dolfin_adjoint

=============================================
Time-dependent optimal control of Stokes flow
=============================================

.. sectionauthor:: Marie E. Rognes <meg@simula.no>,  Steven Vandekerckhove <Steven.Vandekerckhove@kuleuven.be>

******************
Problem definition
******************

The problem is to minimise the following tracking-type functional

.. math::
   J(y, u) =
      \frac{1}{2} \int_{0}^T \int_{\Omega} | y - z |^2 \, \mathrm{d}x \, \, \mathrm{d}t
      + \frac{\alpha}{2} \int_{0}^T \int_{\Omega} |u|^2 \, \mathrm{d}x \, \, \mathrm{d}t

subject to the time-dependent Stokes equation

.. math::
    y_t - \nu \Delta y + \nabla p &= u(t) \qquad \mathrm{in} \, \Omega \times (0, T), \\
    - \nabla \cdot y &= 0 \qquad \mathrm{in} \, \Omega \times (0, T), \\
    y(\cdot, t) &= 0   \qquad \mathrm{on} \, \partial \Omega \times (0, T), \\
    y(\cdot, 0) &= y_0   \qquad \mathrm{in} \, \Omega .

In particular, we aim to

.. math::
   \min J(y, u) \textrm{ over } (y, u)

**************
Discretization
**************

Using the implicit Euler discretization in time with timestep
:math:`\Delta t`, the time-discretized differential equation reads:
for a given :math:`u^n`, for each time step :math:`n`, find
:math:`(y^n, p^n)` such that

.. math::

    y^{n}  - \Delta t \, ( \nu \Delta y^{n} -  \nabla p^{n}) = \Delta t \, u^n + y^{n-1} \\

   - \nabla \cdot y^{n} = 0

Let :math:`V` be the space of continuous piecewise quadratic vector
fields that vanish on the boundary :math:`\partial \Omega`. Let
:math:`Q` be the space of continuous piecewise linear vector functions
that have average value zero. Multiplying by test functions
:math:`\phi \in V` and :math:`q \in Q`, integrating by parts over
:math:`\Omega`, the problem reads: find :math:`y_h^{n} \in V` and
:math:`p_h^{n} \in Q` such that

.. math::

   \langle y_h^{n}, \phi \rangle
   + \Delta t  \langle \nu \nabla y_h^{n}, \nabla \phi \rangle
   - \Delta t \langle p_h^{n}, \nabla \cdot \phi \rangle
      &= \Delta t \langle u^n, \phi \rangle
      + \langle y_h^{n-1}, \phi \rangle \\
   - \Delta t \langle \nabla \cdot y_h^{n}, q \rangle &= 0

hold for all :math:`\phi \in V` and :math:`q \in Q`. Note that we here
have multiplied the last equation by :math:`\Delta t` for the sake of
symmetry: this can be advantageous for the solution of resulting
linear system of equations.

**************
Implementation
**************

We start our implementation by importing the :py:mod:`dolfin` and :py:mod:`dolfin_adjoint` modules:

.. code-block:: python

    from dolfin import *
    from dolfin_adjoint import *
