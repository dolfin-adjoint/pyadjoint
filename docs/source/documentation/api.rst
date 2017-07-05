.. _fenics-adjoint-api-reference:

============================
fenics-adjoint API reference
============================

.. automodule:: fenics_adjoint

********************
Overloaded functions
********************

.. autofunction:: assemble
.. autofunction:: assemble_system
.. autofunction:: solve
.. autofunction:: project

******************
Overloaded objects
******************

.. autoclass:: NonlinearVariationalSolver

   .. automethod:: solve

.. autoclass:: NonlinearVariationalProblem
.. autoclass:: LinearVariationalSolver

   .. automethod:: solve

.. autoclass:: LinearVariationalProblem
.. autoclass:: Function

   .. automethod:: assign
.. autoclass:: Constant

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

.. _parameter-label:

*******************
Debugging functions
*******************
.. autofunction:: visualise
.. autofunction:: taylor_test
