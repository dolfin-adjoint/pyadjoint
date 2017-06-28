.. _dolfin-adjoint-tutorial:

.. py:currentmodule:: dolfin_adjoint

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
fenics_adjoint module **after** loading dolfin:

.. code-block:: python

    from dolfin import *
    from fenics_adjoint import *

The reason why it is necessary to do it afterwards is because
fenics-adjoint overloads many of the dolfin API functions to
understand what the forward code is doing.  In this particular case,
the :py:func:`solve <dolfin_adjoint.solve>` function and
:py:meth:`assign <dolfin_adjoint.Function.assign>` method have been
overloaded:

.. code-block:: python
   :emphasize-lines: 2,3

    while (t <= end):
        solve(F == 0, u_next, bc)
        u.assign(u_next)

The fenics-adjoint versions of these functions will *record* each step
of the model, building an *annotation*, so that it can *symbolically
manipulate* the recorded equations to derive the tangent linear and
adjoint models.  Note that no user code had to be changed: it happens
fully automatically.

In order to talk about adjoints, one needs to consider a particular
functional. While fenics-adjoint supports arbitrary functionals, let
us consider a simple nonlinear example.  Suppose our functional of
interest is the square of the norm of the final velocity:

.. math::

    J(u) = \int_{\Omega} \left\langle u(T), u(T) \right\rangle \ \textrm{d}\Omega,

or in code:

.. code-block:: python

    J = assemble(inner(u, u)*dx),

where u is the final velocity.

..
   |more| If the functional were to be an integral over time, one could
   multiply by :py:data:`*dt`. This requires some more annotation; see
   the documentation for :py:func:`adj_inc_timestep`. For how to express
   more complex functionals, see the documentation on :doc:`expressing
   functionals <functionals>`.


   The dolfin-adjoint software has several drivers, depending on
   precisely what the user requires.  The highest-level interface is to
   compute the gradient of the functional with respect to some
   :py:class:`Control`. For example,

Suppose we wish to compute the
gradient of :math:`J` with respect to the initial condition for
:math:`u`, using the adjoint.  We can do this with the following code:

.. code-block:: python

    dJdic = compute_gradient(J, ic)

where ic is the initial condition for u.
This single function call differentiates the model, assembles each adjoint
equation in turn, and then uses the adjoint solutions to compute the
requested gradient.

..
   Other :py:class:`Controls` are possible. For example, to compute the
   gradient of the functional :math:`J` with respect to the diffusivity
   :math:`\nu`:

If we wish instead to take the gradient with respect to the diffusivity
:math:`\nu`, we can write:
      
.. code-block:: python

    dJdnu = compute_gradient(J, nu)
If we want both gradients we can write
.. code-block:: python

    dJdic, dJdnu = compute_gradient(J,[ic,nu])

Now our whole program is

.. literalinclude:: ../_static/tutorial2.py

Observe how the changes required from the original forward code to the
adjoined version are very small: with only three lines added to the
original code, we are able to compute the gradient information.

|more| If you have been following along, you can `download the
adjoined Burgers' equation code`_ and compare your results.

Other interfaces are available to manually compute the adjoint and
tangent linear solutions. For details, see the section on
:doc:`lower-level interfaces <misc>`.

Once you have computed the gradient, how do you know if it is correct?

..
   If you were to pass an incorrect gradient to an optimisation
   algorithm, the convergence would be hampered or it may fail
   entirely. Therefore, before using any gradients, you should satisfy
   yourself that they are correct.

fenics-adjoint offers easy routines to
rigorously verify the computed results, which is the topic of the
:doc:`next section <verification>`.

.. _their tutorial: http://fenicsproject.org/documentation
.. _download the source code: ../_static/tutorial1.py
.. _download the adjoined Burgers' equation code: ../_static/tutorial2.py

.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
