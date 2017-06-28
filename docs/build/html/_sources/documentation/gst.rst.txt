.. py:currentmodule:: dolfin_adjoint

==============================
Generalised stability analysis
==============================

Generalised stability analysis is a powerful tool for investigating
the stability of physical systems.  For an introduction to the
mathematics, see :doc:`the chapter in the mathematical background
<maths/A-gst>`.

*****************
Computing the GST
*****************

Once the record of the simulation has been created, it is possible to
perform generalised stability analysis with one line of code:

.. code-block:: python

  gst = compute_gst(initial, final, nsv)

The :py:data:`initial` and :py:data:`final` variables define the input
and output of the propagator :math:`L` respectively, while
:py:data:`nsv` is the number of singular vectors requested. By
default, the mass matrices of the input and output spaces are used to
define the norms on the input and output spaces. Optionally, the user
can specify other matrices to change the norms used for either of
these spaces. For example, to replicate the default behaviour, one
could write

.. code-block:: python

  gst = compute_gst(initial, final, nsv, ic_norm=inner(u, v)*dx)

where :py:data:`u` and :py:data:`v` are instances of the
:py:class:`TrialFunction` and :py:class:`TestFunction` classes on the
appropriate input function space.

This one call will derive the tangent linear and adjoint systems,
construct a matrix-free representation of the propagator, and use this
inside a Krylov-Schur iteration to solve the GST singular value
problem. This computation may take many iterations of the tangent
linear and adjoint systems. The solution of the singular value problem
is achieved with the `SLEPc <http://www.grycap.upv.es/slepc/>`_
library.

*************
Using the GST
*************

Once the GST has been computed, it may be used as follows:

.. code-block:: python

  for i in range(gst.ncv):
    (sigma, u, v, residual) = gst.get_gst(i, return_vectors=True, return_residual=True)


The :py:data:`ncv` member of the :py:data:`gst` contains the number of
converged singular vectors. This may be less than, equal to, or
greater than the requested number of singular vectors.

By default, :py:meth:`get_gst` only returns the growth rate
:math:`\sigma` associated with the computed singular triplet. To fetch
the singular vectors, pass :py:data:`return_vectors=True`. To compute
the residual of the eigenvalue computation, pass
:py:data:`return_residual=True`.

*******
Example
*******

A complete example of a generalised stability analysis of the tutorial
example is presented below.

.. literalinclude:: ../_static/tutorial11.py

|more| Download the `gst code`_.

.. _gst code: ../_static/tutorial11.py

This prints the following output:

.. code-block:: none

    $ python tutorial11.py
    ...
    Growth rate of vector 0: 4.09880352035
    Growth rate of vector 1: 3.20037673764
    Growth rate of vector 2: 3.07821571572
    Growth rate of vector 3: 3.06242628866

.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
