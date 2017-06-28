======================
Expressing functionals
======================

In the example presented in the :doc:`tutorial <tutorial>`, the quantity of interest was
evaluated at the end of the simulation. However, it is very common
to want to compute integrals over time, or evaluated at certain points
in time that are not the end. The syntax of the :py:class:`Functional <dolfin_adjoint.Functional>`
class is intended to be very general, and to express all of these needs
naturally.

The core abstraction for functionals in dolfin-adjoint is that a functional is either

a. an integral of a form over a certain time window, or
b. a pointwise evaluation in time of a certain form, or
c. a sum of terms like (a) and (b).

********
Examples
********

To see how it works, consider some examples:

- Integration over all time:

.. code-block:: python

  J = Functional(inner(u, u)*dx*dt)

- Integration over a certain time window:

.. code-block:: python

  J = Functional(inner(u, u)*dx*dt[0:1])

- Integration from a certain point until the end:

.. code-block:: python

  J = Functional(inner(u, u)*dx*dt[0.5:])

- Pointwise evaluation in time (does not need to line up with timesteps):

.. code-block:: python

  J = Functional(inner(u, u)*dx*dt[0.5])

- Pointwise evaluation at the start (e.g. for regularisation terms):

.. code-block:: python

  J = Functional(inner(u, u)*dx*dt[START_TIME])

- Pointwise evaluation at the end:

.. code-block:: python

  J = Functional(inner(u, u)*dx*dt[FINISH_TIME])

- And sums of these work too:

.. code-block:: python

  J = Functional(inner(u, u)*dx*dt + inner(u, u)*dx*dt[FINISH_TIME])

The object to express these evaluations in time is the :py:class:`TimeMeasure <dolfin_adjoint.TimeMeasure>`
object. By default, dolfin-adjoint creates a :py:data:`TimeMeasure` called :py:data:`dt`. If your code
redefines :py:data:`dt`, you will need to instantiate a new :py:data:`TimeMeasure` class under a different
name:

.. code-block:: python

  # Can't use dt, because it's been set to be the timestep
  dtm = TimeMeasure()
  J = Functional(inner(u, u)*dx*dtm + inner(u, u)*dx*dtm[FINISH_TIME])

***********
Limitations
***********

If you use a complicated functional, you should be aware of some points.

1. In order for anything other than evaluation at the end of time to work,
   you need to call :py:func:`adj_start_timestep <dolfin_adjoint.adj_start_timestep>` before
   your time loop, and :py:func:`adj_inc_timestep <dolfin_adjoint.adj_inc_timestep>`
   function as the last instruction in your time loop. This is to make dolfin-adjoint aware that a timestep has ended,
   and the start and end times of that timestep.
2. Evaluation of expressions at times other than timesteps is currently performed using
   linear interpolation, which may not be the correct thing to do if you are
   using a higher-order scheme in time.
