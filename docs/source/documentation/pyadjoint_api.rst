.. _pyadjoint-api-reference:

============================
pyadjoint API reference
============================

See also the :doc:`dolfin-adjoint API reference <api>`.

.. automodule:: pyadjoint



******************
Overloaded objects
******************

.. autoclass:: AdjFloat
.. autoclass:: OverloadedType


**********
Annotation
**********

.. autoclass:: Block

   .. automethod:: __init__

.. autoclass:: Tape

   .. automethod:: visualise

****************
Driver functions
****************

.. autofunction:: compute_gradient


***********************************
:py:data:`ReducedFunctional` object
***********************************

.. autoclass:: ReducedFunctional

   .. automethod:: __call__
   .. automethod:: derivative

************
Verification
************

.. autofunction:: taylor_test
