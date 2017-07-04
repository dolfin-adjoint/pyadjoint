======================
Expressing functionals
======================

In the example presented in the :doc:`tutorial <tutorial>`, the quantity of interest was
evaluated at the end of the simulation. However, it is very common
to want to compute integrals over time, or evaluated at certain points
in time that are not the end. With fenics-adjoint this is very straightforward.


********
Examples
********

To see how it works, we once again consider the Burgers equation example
from the tutorial:

.. literalinclude:: ../_static/tutorial2.py

Here the functional considered was

.. math::

   J(u) = \int_{\Omega} \left\langle u(T), u(T) \right\rangle \ \textrm{d}\Omega.

Let us see how we have to change the program to accomedate different functionals:
                    

- Integration over all time:

  .. math::
   
     J(u) = \int_0^T\int_{\Omega}\left\langle u(t),u(t)\right\rangle \ \textrm{d}\Omega \ \textrm{d}t

  We need to perform the integral numerically.
  To do this we should change the forward code to compute the time independent part of :math:`J`
  at each time step and save the value to a list:

  .. code-block:: python

     Jlist = []

     t = 0.0
     end = 0.1
     while (t <= end):
         solve(F == 0, u_next, bc)
         u.assign(u_next)
         Jtemp = assemble(inner(u,u)*dx)
         Jlist.append([t,Jtemp])
         t += float(timestep)

  Now we can integrate up :math:`J` for example by the trapezoidal rule:

  .. code-block:: python

     J = 0
     for i in range(1, len(Jlist)):
         J += (Jlist[i-1][1] + Jlist[i][1])*0.5*float(timestep)

  
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
