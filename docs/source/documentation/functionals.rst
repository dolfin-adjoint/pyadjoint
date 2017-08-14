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

Let us see how we have to change the program to accomedate different functionals with different time dependencies:
To do this we should change the forward code to compute part of :math:`J`
at each time step and save the value to a list:

.. code-block:: python
   :emphasize-lines: 3, 4, 10, 11

   t = 0.0
   end = 0.1
   Jtemp = assemble(inner(u,u)*dx)
   Jlist = [Jtemp]
   while (t <= end):
       solve(F == 0, u_next, bc)
       u.assign(u_next)
       t += float(timestep)

       Jtemp = assemble(inner(u, u)*dx)
       Jlist.append(Jtemp)


Let us look at some specific functionals:

- Integration over all time:

  .. math::

     J(u) = \int_0^T\int_{\Omega}\left\langle u(t),u(t)\right\rangle \ \textrm{d}\Omega \ \textrm{d}t

  We need to perform the time integral numerically, for example by the trapezoidal rule:

  .. code-block:: python

     J = 0
     for i in range(1, len(Jlist)):
         J += 0.5*(Jlist[i-1] + Jlist[i])*float(timestep)

  We could also use ready-made integration routines, but we have to make sure that the routine does
  not change the type of the :py:data:`J`. :py:data:`Jtemp` and :py:data:`J` have
  type :py:class:`AdjFloat <pyadjoint.AdjFloat>`.

  ..
     For example if we wanted to use the scipy
     trapezoidal rule function we could write

     .. code-block:: python

        from scipy.integrate import trapz
        from numpy import array

        Jlist = array(Jlist, dtype=AdjFloat)
        J = trapz(Jlist, dx=float(timestep))

  |more| Download the `code to find the full time integral`_.

  .. _code to find the full time integral: ../_static/tutorial8.py

- Integration over a certain time window:

  .. math::

     J(u) = \int_{t_1}^{t_2}\int_{\Omega}\left\langle u(t),u(t)\right\rangle \ \textrm{d}\Omega \ \textrm{d}t

  We can again use the trapezoidal rule. Compared to the full time integration we only have to change the looping range.
  If we use our example with :math:`t_{1} = 0.03` and
  :math:`t_{2} = 0.07`, then we can write

  .. code-block:: python

     J = 0
     for i in range(4, 8):
         J += 0.5*(Jlist[i-1] + Jlist[i])*float(timestep)


- Integration from a certain point until the end:

  .. math::

     J(u) = \int_{t_1}^{T}\int_{\Omega}\left\langle u(t),u(t)\right\rangle \ \textrm{d}\Omega \ \textrm{d}t

  Again we only change the loop range. If we use our example with :math:`t_{1} = 0.03` we can write

  .. code-block:: python

     J = 0
     for i in range(4,len(Jlist)):
         J += 0.5*(Jlist[i-1] + Jlist[i])*float(timestep)


- Pointwise evaluation in time:

  .. math::

     J(u) = \int_{\Omega}\left\langle u(t_1),u(t_1)\right\rangle \ \textrm{d}\Omega

  Here we only need to pick out the functional from the list, for example if :math:`t_1 = 0.03`:

  .. code-block:: python

     J = Jlist[3]


- Pointwise evaluation at the start (e.g. for regularisation terms):

  .. math::

     J(u) = \int_{\Omega}\left\langle u(0),u(0)\right\rangle \ \textrm{d}\Omega

  Again we only need to pick out the functional from the list:

  .. code-block:: python

     J = Jlist[0]


- Pointwise evaluation at the end:

  .. math::

     J(u) = \int_{\Omega}\left\langle u(T),u(T)\right\rangle \ \textrm{d}\Omega

  Here we only need to pick out the functional from the list:

  .. code-block:: python

     J = Jlist[-1]


- And sums of these work too:

  .. math::

     J(u) = \int_0^T\int_{\Omega}\left\langle u(t),u(t)\right\rangle \ \textrm{d}\Omega \ \textrm{d}t  + \int_{\Omega}\left\langle u(T),u(T)\right\rangle \ \textrm{d}\Omega



  .. code-block:: python

     J = 0
     for i in range(1, len(Jlist)):
         J += 0.5*(Jlist[i-1] + Jlist[i])*float(timestep)
     J += JList[-1]

- Ratio of evaluation at different times

  .. math::

     J(u) = \frac{\int_{\Omega}\left\langle u(t_2),u(t_2)\right\rangle \ \textrm{d}\Omega}{\int_{\Omega}\left\langle u(t_1),u(t_1)\right\rangle \ \textrm{d}\Omega}

  for example with :math:`t_1 = 0` and :math:`t_2 = 0.03`:

  .. code-block:: python

     J = Jlist[3]*Jlist[0]**(-1)


In the :doc:`next section <custom_functions>` we discuss how to use pyadjoint with functions other than FEniCS functions.


.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
