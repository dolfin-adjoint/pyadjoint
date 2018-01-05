.. _dolfin-adjoint-api-reference:

============================
dolfin-adjoint API reference
============================

See also the :doc:`pyadjoint API reference <pyadjoint_api>`.

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
.. autoclass:: DirichletBC
.. autoclass:: Expression
