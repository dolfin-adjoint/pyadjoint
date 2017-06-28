.. py:currentmodule:: dolfin_adjoint

=================================================================
Time-dependent optimal control of the linear scalar wave equation
=================================================================

.. sectionauthor:: Steven Vandekerckhove

******************
Problem definition
******************

The problem is to minimise the following tracking-type functional

.. math::
   J(y, u) =
      \frac{1}{2} \int_{0}^T | u(L, t) - u_\text{obs}(L, t) |^2 \, \, \mathrm{d}t,

subjected to the time-dependent scalar wave equation equation

.. math::
    u_{tt} - c^2 u_{xx} &= 0 \qquad \mathrm{in} \, \Omega \times (0, T), \\
    u(x, 0) &= 0, \\
    u(0, t) &= s(t), \\
    u(L, t) &= 0,

where :math:`c` is the wave speed and :math:`\Omega = \left[0, L\right]` is a one dimensional domain,  for a given source function :math:`s(t) = \sin(\omega t)`:

In particular, we aim to

.. math::
   \min J(u, \omega) \textrm{ over } (u, \omega).

**************
Discretization
**************

Using a classic central difference for discretizing in time, with time step
:math:`\Delta t`, the time-discretized differential equation reads:
for a given :math:`s^n`, for each time step :math:`n`, find
:math:`u^{n+1}` such that

.. math::

    \frac{u^{n+1} - 2 u^n + u^{n-1}}{\Delta t^2} - c^2 u^n_{xx} &= 0, \\

   u(0, t^n) = s(t^n) &= s^n.

Let :math:`U` be the space of continuous piecewise linear functions.
Multiplying by test functions :math:`v \in U`, integrating by parts over
:math:`\Omega`, the problem reads: find :math:`u_h^{n} \in U` such that

.. math::


   \langle \frac{u^{n+1} - 2 u^n + u^{n-1}}{\Delta t^2}, v \rangle
   + \langle c^2 u^n_x, v_x \rangle &= 0,

hold for all :math:`v \in U`.

**************
Implementation
**************

We start our implementation by importing the :py:mod:`dolfin` and :py:mod:`dolfin_adjoint` modules,
together with the numpy and sys modules, for handeling storage and ui:

.. literalinclude:: ../../_static/time-dependent-wave.py
   :lines: 1-4

Next, we prepare the mesh,

.. literalinclude:: ../../_static/time-dependent-wave.py
   :lines: 10

and set a time step size:

.. literalinclude:: ../../_static/time-dependent-wave.py
   :lines: 13

Since we want to add boundary conditions only on the left hand side,
and make observations on the left hand side, we have to identify both sides
separately:

.. literalinclude:: ../../_static/time-dependent-wave.py
   :lines: 15-23

Then, an expression is built for the time dependent source term.
We need to provide member functions for evaluating the function and its derivative.

.. literalinclude:: ../../_static/time-dependent-wave.py
   :lines: 25-45

Before the inverse problem can be solved, we have to implement the forward problem:

.. literalinclude:: ../../_static/time-dependent-wave.py
   :lines: 46-94

Note that the forward solver has been implemented as straight forward as possible,
with litte attention for efficiency. E.g., a significant speed-up could be realized
by re-using the factorization of linear system.

Also a function is defined to assemble the objective

.. literalinclude:: ../../_static/time-dependent-wave.py
   :lines: 104-112

Now we can have a look at the optimization procedure

.. literalinclude:: ../../_static/time-dependent-wave.py
   :lines: 114-167

The code can be run as follows:

.. code-block:: python

    """ Compute  a reference solution (once) """
    Source = source(omega = Constant(2e2), degree=3)
    forward(Source, 2*DOLFIN_PI, True)

    """ Start the optimization procedure """
    optimize()

The complete code can be downloaded `here <../../_static/time-dependent-wave.py>`_.

********
Comments
********

Running the code results in an approximation for the optimal value which is correct up to the noise level will be obtained.
