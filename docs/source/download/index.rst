.. _download:

*************************
Installing dolfin-adjoint
*************************

Make sure that you have `FEniCS`_ installed, see `their installation instructions`_.

Then install dolfin-adjoint with:

    pip install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@master

Test your installation by running:
    python -m "import fenics_adjoint"

Optional dependencies:
----------------------

- `IPOPT`_ and `pyipopt`_: This is the best available open-source optimisation algorithm. Strongly recommended if you wish to solve :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Make sure to compile IPOPT against the `Harwell Subroutine Library`_.

- `Moola`_: A set of optimisation algorithms specifically designed for :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`.

- `Optizelle`_: An Open Source Software Library Designed To Solve General Purpose Nonlinear Optimization Problems.

.. _FEniCS: http://fenicsproject.org
.. _Optizelle: http://www.optimojoe.com/products/optizelle
.. _SLEPc: http://www.grycap.upv.es/slepc/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _pyipopt: https://github.com/xuy/pyipopt
.. _moola: https://github.com/funsim/moola
.. _Harwell Subroutine Library: http://www.hsl.rl.ac.uk/ipopt/
.. _their installation instructions: http://fenicsproject.org/download
