.. py:currentmodule:: dolfin_adjoint

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
The main function in fenics-adjoint for applying the Taylor remainder convergence test is :py:func:`taylor_test <dolfin_adjoint.taylor_test>`.
To see how this works, let's restructure our forward model so that we can run it as a pure function of
the diffusivity :math:`\nu`:

.. literalinclude:: ../_static/tutorial3.py

As you can see, we've taken the action part of the model into a function :py:data:`main`, so that we
can drive the forward model several times for verification. Now let's see how to use :py:func:`taylor_test <dolfin_adjoint.taylor_test>`:

.. literalinclude:: ../_static/tutorial4.py

The h must be the same type as what we are differentiating with respect to, so in this case since nu is a constant h must be a constant.
Also note that calling Jhat with a new constant for example Jhat(nu + h) changes nu to nu + h. For this reason we pass Constant(nu) to the taylor test rather than nu.
If we were differentiating with respect to something other than a function, for example the function u we could do

.. code-block:: python
   h = Function(V)
   h.vector()[:] = 1
   taylor_test(RestrictedFunctional(J,u),u.copy(deepcopy=True),h)

|more| Download the `adjoint code with verification`_.

.. _adjoint code with verification: ../_static/tutorial4.py


Running this program yields the following output (well, not really!!!):

.. code-block:: none

    $ python tutorial4.py
    ...
    Taylor remainder without gradient information: 
      [0.0023634768826859553, 0.001219686435181555, 0.0006197555788530762, 
      0.0003124116082189321, 0.0001568463925042951]

The first line gives the values computed for the first-order Taylor remainder. As you can see, each value is approximately half the previous one. 

.. code-block:: none

    Convergence orders for Taylor remainder without gradient information (should all be 1): 
      [0.9544004555219242, 0.9767390399645689, 0.9882512926546192, 0.9940957131087177]

The second line shows the convergence orders of the first-order Taylor remainders: these should
always be 1. (If they are not, try decreasing h to
use a smaller perturbation.) 

.. code-block:: none

    Taylor remainder with gradient information: 
      [0.00015639195264909554, 4.0247982485970384e-05, 1.0211629980686528e-05, 
      2.5719961979492594e-06, 6.454097041455739e-07]

The third line gives the values computed for the second-order Taylor remainder. These values should be much smaller than those on
the first line. 

.. code-block:: none

    Convergence orders for Taylor remainder with gradient information (should all be 2): 
      [1.9581779067535698, 1.9787032993204938, 1.9892527525050359, 1.9946013350664813]

The fourth line shows the convergence orders of the second-order Taylor remainders: if the gradient has been computed correctly with the adjoint, then
these numbers should be 2.

As can be seen, the second-order Taylor remainders do indeed converge at second order, and so the gradient :py:data:`dJdnu` is correct.

So, what if the Taylor remainders are not correct? Such a situation could occur if the model
manually modifies :py:class:`Function <dolfin_adjoint.Function>` values, or if the model modifies the entries of assembled matrices and
vectors, or if the model is not differentiable, or if there is a bug in fenics-adjoint. fenics-adjoint offers many ways to pinpoint
precisely where an error might lie; these are discussed in the :doc:`next section on debugging
<debugging>`.

..
   Once the adjoint model development is completed, you may wish to run your model on much bigger
   production runs. **By default, dolfin-adjoint stores all variables computed in memory**, as they
   will be necessary to linearise the model about the forward trajectory. If you wish to solve much
   bigger problems, or if the model must run for many timesteps, it may not be feasible to store
   everything: some balance between storage and recomputation must be achieved, in the form of a
   **checkpointing scheme**. Checkpointing is the topic of the :doc:`next section <checkpointing>`.

.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
