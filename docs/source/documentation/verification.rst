.. py:currentmodule:: fenics_adjoint

============
Verification
============

*********************************
Taylor remainder convergence test
*********************************

The fundamental tool used in verification of gradients is the
*Taylor remainder convergence test*. Let :math:`\widehat{J}(m)` be the functional, considered
as a pure function of the parameter of interest,
let :math:`\nabla \widehat{J}` be its gradient, and let :math:`\delta m` be a perturbation to
:math:`m`. This test is based on the observation that

.. math::

    \left|\widehat{J}(m + h\delta m) - \widehat{J}(m)\right| \rightarrow 0 \quad \mathrm{at} \ O(h),

but that

.. math::

    \left|\widehat{J}(m + h\delta m) - \widehat{J}(m) - h\nabla \widehat{J} \cdot \delta m\right| \rightarrow 0 \quad \mathrm{at} \ O(h^2),

by Taylor's theorem. The quantity :math:`\left|\widehat{J}(m + h\delta m) - \widehat{J}(m)\right|` is called the *first-order
Taylor remainder* (because it's supposed to be first-order), and the quantity :math:`\left|\widehat{J}(m + h\delta m) - \widehat{J}(m) - h\nabla \widehat{J} \cdot \delta m\right|`
is called the *second-order Taylor remainder*.

Suppose someone gives you two functions :math:`\widehat{J}` and :math:`d\widehat{J}`, and claims that :math:`d\widehat{J}` is the gradient of
:math:`\widehat{J}`. Then their claim can be rigorously verified by computing the second-order Taylor remainder for some
choice of :math:`h` and :math:`\delta m`, then repeatedly halving :math:`h` and checking that the result decreases
by a factor of 4.

*******************************
Applying this in fenics-adjoint
*******************************

In the case of PDE-constrained optimisation, computing :math:`\widehat{J}(m)` involves solving the PDE
for that choice of :math:`m` to compute the solution :math:`u`, and then evaluating the functional :math:`J`.
The main function in fenics-adjoint for applying the Taylor remainder convergence test is :py:func:`taylor_test <fenics_adjoint.taylor_test>`.
To see how this works, let us again consider our example with Burgers' equation :math:`\nu`:

.. literalinclude:: ../_static/tutorial3.py

As you can see, we here find the gradient only with respect to :py:data:`nu`.
Now let's see how to use :py:func:`taylor_test <fenics_adjoint.taylor_test>`:
Instead of

.. code-block:: python

   dJdnu = compute_gradient(J, nu)

we write

.. code-block:: python

   h = Constant(0.0001)
   Jhat = ReducedFunctional(J, nu)
   conv_rate = taylor_test(Jhat, nu, h)

Here, :py:data:`h` is the direction of perturbation.
:py:data:`h` must be the same type as what we are differentiating with respect to, so in this case since :py:data:`nu` is a :py:class:`Constant <fenics_adjoint.Constant>` :py:data:`h` must also be a :py:class:`Constant <fenics_adjoint.Constant>`.
It is also a good idea to make sure that :py:data:`h` is the same order of magnitude as :py:data:`nu`, so that the perturbations are not too large.
:py:data:`Jhat` is the functional reduced to a pure function of :py:data:`nu`.
We could also have taken the taylor test on the gradient with respect to the :py:class:`Function <fenics_adjoint.Function>`
:py:data:`u`. In that case :py:data:`h` must also be a :py:class:`Function <fenics_adjoint.Function>`.

.. code-block:: python

   h = Function(V)
   h.vector()[:] = 0.1
   conv_rate = taylor_test(RestrictedFunctional(J, u), u, h)

Here is the full program to check that we compute :py:data:`dJdnu` correctly:

.. literalinclude:: ../_static/tutorial4.py

|more| Download the `adjoint code with verification`_.

.. _adjoint code with verification: ../_static/tutorial4.py


Running this program yields the following output:

.. code-block:: none

    $ python tutorial4.py
    ...
    Computed residuals: [8.7896393952526051e-07, 2.2008124772799524e-07, 5.5062930799269294e-08, 1.3771065357994394e-08]
    Computed convergence rates: [1.9977677544105585, 1.9988829175084986, 1.9994412277283045]


The first line gives the values computed for the second-order Taylor remainder. As you can see, each value is approximately one quarter of the previous one.
The second line gives the convergence orders of the second-order Taylor remainder: if the gradient has been computed correctly these numbers should be 2.
As we can see they are in fact very close to 2, so we are calculating the gradient correctly.

If you want to see if some object is the gradient you can pass the inner product of that object and the direction :py:data:`h` with the named argument :py:data:`dJdm`.
For example we may want to check that the convergence orders of the first-order Taylor remainder are 1. This is achieved by passing a proposed gradient 0:

.. code-block:: python

    conv_rate = taylor_test(Jhat, Constant(nu), h, dJdm = 0)

Adding this we get the output

.. code-block:: none

   $ python tutorial4.py
   ...
   Computed residuals: [0.00025403832691939243, 0.00012723856418173085, 6.367425978393015e-05, 3.185089029200672e-05]
   Computed convergence rates: [0.99751017666093167, 0.99875380873361586, 0.99937658413144936]

We see that the residuals are halved and the convergence rates are 1 as expected.

So, what if the Taylor remainders are not correct? Such a situation could occur if the model
manually modifies :py:class:`Function <fenics_adjoint.Function>` values, or if the model modifies the entries of assembled matrices and
vectors, or if the model is not differentiable, or if there is a bug in fenics-adjoint. fenics-adjoint offers ways to pinpoint
precisely where an error might lie; these are discussed in the :doc:`next section on debugging
<debugging>`.


.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
