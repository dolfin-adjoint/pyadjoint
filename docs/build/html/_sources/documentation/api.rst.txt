.. _dolfin-adjoint-api-reference:

============================
dolfin-adjoint API reference
============================

.. automodule:: dolfin_adjoint

********************
Overloaded functions
********************

.. autofunction:: assemble
.. autofunction:: assemble_system
.. autofunction:: solve
.. autofunction:: project
.. autofunction:: interpolate

******************
Overloaded objects
******************

.. autoclass:: LUSolver

   .. automethod:: solve

.. autoclass:: NewtonSolver

   .. automethod:: solve

.. autoclass:: KrylovSolver

   .. automethod:: solve

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
.. autofunction:: compute_adjoint
.. autofunction:: compute_tlm

*****************************
:py:data:`Functional` object
*****************************

.. autoclass:: Functional

***********************************
:py:data:`ReducedFunctional` object
***********************************

.. autoclass:: ReducedFunctional

   .. automethod:: __call__
   .. automethod:: derivative
   .. automethod:: hessian
   .. automethod:: taylor_test

.. autoclass:: ReducedFunctionalNumPy

   .. automethod:: __call__
   .. automethod:: derivative
   .. automethod:: hessian
   .. automethod:: pyopt_problem
   .. automethod:: set_controls
   .. automethod:: get_controls

.. _parameter-label:

****************************
:py:data:`Control` objects
****************************

.. autofunction:: Control
.. autoclass:: FunctionControl
.. autoclass:: ConstantControl

****************************
Constraint objects
****************************

.. autoclass:: EqualityConstraint

   .. automethod:: function
   .. automethod:: jacobian

.. autoclass:: InequalityConstraint

   .. automethod:: function
   .. automethod:: jacobian

********************
Annotation functions
********************

.. autofunction:: adj_checkpointing
.. autofunction:: adj_start_timestep
.. autofunction:: adj_inc_timestep

*******************
Debugging functions
*******************

.. autofunction:: adj_html
.. autofunction:: adj_check_checkpoints
.. autofunction:: taylor_test
.. autofunction:: replay_dolfin

****************************
Generalised stability theory
****************************

.. autofunction:: compute_gst

***********************************
Accessing tape
***********************************

.. autoclass:: DolfinAdjointVariable

   .. automethod:: __init__
   .. automethod:: tape_value
   .. automethod:: iteration_count
   .. automethod:: known_timesteps


.. autofunction:: adj_reset
