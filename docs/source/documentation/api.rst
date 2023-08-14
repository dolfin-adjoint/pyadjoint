:orphan:

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
.. autofunction:: interpolate

******************
Overloaded objects
******************

.. autoclass:: KrylovSolver

    .. automethod:: solve
.. autoclass:: LUSolver

    .. automethod:: solve

.. autoclass:: NewtonSolver

    .. automethod:: solve
.. autoclass:: NonlinearVariationalSolver

   .. automethod:: solve

.. autoclass:: NonlinearVariationalProblem
.. autoclass:: LinearVariationalSolver

   .. automethod:: solve

.. autoclass:: LinearVariationalProblem

****************
Overloaded types
****************

.. autoclass:: Constant
.. autoclass:: DirichletBC
.. autoclass:: Expression
.. autoclass:: Function

    .. automethod:: assign
