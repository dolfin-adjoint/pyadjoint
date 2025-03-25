:orphan:

.. _download:

*************************
Installing pyadjoint
*************************

Firedrake
---------
 
Pyadjoint is automatically installed in Firedrake. No separate
installation is required.

PIP (all platforms)
================================

Install dolfin-adjoint and its Python dependencies with pip:

.. code-block:: bash

    pip install git+https://github.com/dolfin-adjoint/pyadjoint.git


Optional dependencies:
----------------------

- `IPOPT`_ and Python bindings (`cyipopt`_): This is the best available open-source optimisation algorithm. Strongly recommended if you wish to solve :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Make sure to compile IPOPT against the `Harwell Subroutine Library`_.

- `Moola`_: A set of optimisation algorithms specifically designed for :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`.

- `Optizelle`_: An Open Source Software Library Designed To Solve General Purpose Nonlinear Optimization Problems.

.. _Optizelle: http://www.optimojoe.com/products/optizelle
.. _SLEPc: http://www.grycap.upv.es/slepc/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _cyipopt: https://github.com/matthias-k/cyipopt
.. _moola: https://github.com/funsim/moola
.. _Harwell Subroutine Library: http://www.hsl.rl.ac.uk/ipopt/
.. _their installation instructions: http://fenicsproject.org/download


Source code
===========

The source code of `pyadjoint` is available on https://github.com/dolfin-adjoint/pyadjoint.
