.. _dolfin-adjoint-tutorial:

.. py:currentmodule:: fenics_adjoint

===========
First steps
===========

********
Foreword
********

If you have never used the FEniCS system before, you should first read
`their tutorial`_.  If you're not familiar with adjoints and their
uses, see the :doc:`background <maths/index>`.

***************
A first example
***************

Let's suppose you are interested in solving the nonlinear
time-dependent Burgers equation:

.. math::

    \frac{\partial \vec u}{\partial t} - \nu \nabla^2 \vec u + \vec u \cdot \nabla \vec u = 0,

subject to some initial and boundary conditions.


A forward model that solves this problem with P2 finite elements might
look as follows:

.. literalinclude:: ../_static/tutorial1.py

|more| You can `download the source code`_ and follow along as we
adjoin this code.

The first change necessary to adjoin this code is to import the
dolfin-adjoint module **after** loading FEniCS:

.. code-block:: python

    from fenics import *
    from fenics_adjoint import *

The reason why it is necessary to do it afterwards is because
dolfin-adjoint overloads many of the dolfin API functions to
understand what the forward code is doing.  In this particular case,
the :py:func:`solve <fenics_adjoint.solve>` function and
:py:meth:`assign <fenics_adjoint.Function.assign>` method have been
overloaded:

.. code-block:: python
   :emphasize-lines: 2,3

    while (t <= end):
        solve(F == 0, u_next, bc)
        u.assign(u_next)

The dolfin-adjoint versions of these functions will *record* each step
of the model, building an *annotation*, so that it can *symbolically
manipulate* the recorded equations to derive the tangent linear and
adjoint models.  Note that no user code had to be changed: it happens
fully automatically.

In order to talk about adjoints, one needs to consider a particular
functional. While dolfin-adjoint supports arbitrary functionals, let
us consider a simple nonlinear example.  Suppose our functional of
interest is the square of the norm of the final velocity:

.. math::

    J(u) = \int_{\Omega} \left\langle u(T), u(T) \right\rangle \ \textrm{d}\Omega,

or in code:

.. code-block:: python

    J = assemble(inner(u, u)*dx),

where :py:data:`u` is the final velocity.

Suppose we wish to compute the
gradient of :math:`J` with respect to the initial condition for
:math:`u`, using the adjoint. Then since :py:data:`u` is updated over time
we must tell dolfin-adjoint that we are interested in the initial value of :py:data`u` by adding:

.. code-block:: python

    control = Control(u)

after initializing :py:data:`u`.
We can then add the following code to compute the gradient:

.. code-block:: python

    dJdu = compute_gradient(J, control)


This single function call differentiates the model, assembles each adjoint
equation in turn, and then uses the adjoint solutions to compute the
requested gradient.

If we wish instead to take the gradient with respect to the diffusivity
:math:`\nu`, we can write:

.. code-block:: python

    dJdnu = compute_gradient(J, Control(nu))

If we want both gradients we can write

.. code-block:: python

    dJdu, dJdnu = compute_gradient(J, [control, Control(nu)])

Now our whole program is

.. literalinclude:: ../_static/tutorial2.py

Observe how the changes required from the original forward code to the
adjoined version are very small: with only three lines added to the
original code, we are able to compute the gradient information.

|more| If you have been following along, you can `download the
adjoined Burgers' equation code`_ and compare your results.


Once you have computed the gradient, how do you know if it is correct?


dolfin-adjoint offers easy routines to
rigorously verify the computed results, which is the topic of the
:doc:`next section <verification>`.

.. _their tutorial: http://fenicsproject.org/documentation
.. _download the source code: ../_static/tutorial1.py
.. _download the adjoined Burgers' equation code: ../_static/tutorial2.py

.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
