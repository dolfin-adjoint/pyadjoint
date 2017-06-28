.. py:currentmodule:: dolfin_adjoint

=========
Debugging
=========

dolfin-adjoint offers a thorough suite of debugging routines, to identify exactly why the adjoint might not be
correct.

*******************************************
Visualising the forward and adjoint systems
*******************************************

It is sometimes useful when debugging a problem to see dolfin-adjoint's interpretation of your forward system,
and the other models it derives from that. The :py:func:`adj_html <dolfin_adjoint.adj_html>` function dumps a HTML visualisation:

.. code-block:: python

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

For example, let us include these in the Burgers' equation example:

.. literalinclude:: ../_static/tutorial7.py
   :emphasize-lines: 41,42

|more| Download the `code to dump the HTML visualisation`_.

.. _code to dump the HTML visualisation: ../_static/tutorial7.py

The resulting `forward`_ and `adjoint`_ tables are available for inspection.

Each row corresponds to one equation solve. The variable being solved for is listed
at the top. If the variable is green, the value of that variable is recorded; if the
variable is red, the value of that variable is not recorded. To identify the dependencies
of each operator, hover over the block (on the diagonal and on the rhs) with the mouse.

.. _forward: ../_static/forward.html
.. _adjoint: ../_static/adjoint.html


*************************
Replaying the forward run
*************************

In order to derive a consistent adjoint model, dolfin-adjoint must correctly understand
your forward model. If dolfin-adjoint's record of your forward model is incorrect, then it
cannot derive a correct adjoint model.

One way this could happen is if the forward model manually modifies the :py:data:`.vector()`
of a :py:class:`Function <dolfin_adjoint.Function>`. For example, suppose that instead of using

.. code-block:: python

    u.assign(u_next)

the code used

.. code-block:: python

    u.vector()[:] = u_next.vector()

then the adjoint would be incorrect, as dolfin-adjoint cannot detect the modification:

.. code-block:: none

    $ python tutorial_incorrect.py
    ...
    Convergence orders for Taylor remainder with adjoint information (should all be 2): 
      [0.9544004555220237, 0.9767390399643741, 0.9882512926547484, 0.9940957131097388]

How would we detect this situation? To check the *consistency* of dolfin-adjoint's annotation,
it can **replay its interpretation of the forward model and compare the results to the real
forward model**. To do this, use the :py:func:`replay_dolfin <dolfin_adjoint.replay_dolfin>` function:

.. code-block:: python

    success = replay_dolfin(tol=0.0, stop=True)

To see this in action, consider the following (**incorrect**) code:

.. literalinclude:: ../_static/tutorial8.py
    :emphasize-lines: 26,35

|more| Download the `broken adjoint code, where the error is detected by replay`_.

.. _broken adjoint code, where the error is detected by replay: ../_static/tutorial8.py

The replay detects that the interpretation and the real forward model diverge:

.. code-block:: none
    
    $ python tutorial8.py
    ...
    Comparing w_4:1:0:Forward against previously recorded value: 
       norm of the difference is 7.120735e-02 (> tolerance of 0.000000e+00)

:py:data:`w_4` refers to :py:data:`u_next` (try printing it), and :py:data:`w_4:1:0` means
"the function w_4 at timestep 1 and iteration 0". In this error message, libadjoint
tells us that the second solution of :py:data:`u_next` is different in the replay than in the forward model.
Of course, this is because dolfin-adjoint's knowledge of the value of :py:data:`u` is wrong. With this
information, the user can inspect the code around the solution for :py:data:`u_next` and examine
more closely.

Note that the replay cannot be exactly perfect when the model is run in parallel:
the order of the parallel reductions is nondeterministic, and so the answers can diverge
within floating-point roundoff. When debugging, run in serial.

************************************
Testing the derivatives of operators
************************************

If the replay works perfectly, but the adjoint is still incorrect, then there are very
few possibilities for what could be wrong. The only other major possibility is if the
model uses a *discretisation that is not differentiable*. In order to assemble the
adjoint equations, any operators in the forward model that depend on previously computed values
must be differentiated with respect to those values. If that dependency is not differentiable,
then no consistent derivative of the model exists.

A simple way to check the differentiability of your model is to set

.. code-block:: python

    parameters["adjoint"]["test_derivative"] = True

before solving any equations. Then, whenever libadjoint goes to assemble a term
involving the derivative of a nonlinear operator, it will apply the :doc:`Taylor test <verification>`
(at the level of the operator, instead of the whole model). For example, a typical error message
from the derivative test looks like

.. code-block:: python

    >>>> 
    Traceback (most recent call last):
    ...
    libadjoint.exceptions.LibadjointWarnComparisonFailed: Expected the Taylor series remainder 
       of operator 96c04e026b91e44576aa43ccef66e6a8 with respect to w_{16}:1:22:Forward to 
       converge at second order, but got 1.000000 
       (the error values are 1.393963e-11 and 6.969817e-12).

In this message, libadjoint tells us that the operator "96c04e026b91e44576aa43ccef66e6a8" (the names are automatically generated from the hash of the form) depends on the variable
w_{16}:1:22, but that its dependence is not differentiable (the Taylor remainder convergence test yielded 1, instead of 2). In this example, the operator is an upwinded DG
discretisation of the advection operator, and w_{16}:1:22 is an advecting velocity.

Note that even if the adjoint is not perfectly consistent (i.e., the Taylor remainders do not converge at second order), the resulting
gradients can still be "good enough" for the purposes of an optimisation algorithm. All that matters to the optimisation algorithm is
that the gradient provides a descent direction; if the Taylor remainders are "small", then the convergence of the algorithm will usually
not be affected. Thus, **the adjoint is generally still useful, even for nondifferentiable discretisations**.

Note that a more rigorous approach for the case where the functional is nondifferentiable is to consider the functional gradient produced by the adjoint as a
*subgradient*. For more information, see `the Wikipedia`_.

.. _the Wikipedia: http://en.wikipedia.org/wiki/Subgradient_method


.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info

