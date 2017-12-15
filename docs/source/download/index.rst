.. _download:

*************************
Installing dolfin-adjoint
*************************

Using pip
=========
   
Make sure that you have `FEniCS`_ installed, see `their installation instructions`_.

Then installing dolfin-adjoint with:

    pip install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@master

Test your installation by running:
    python -m "import fenics_adjoint"


Dependencies
============

- `FEniCS`_. For installation instructions for FEniCS, see `their installation instructions`_.

Optional dependencies:
----------------------

- `SLEPc`_. This is necessary if you want to conduct :doc:`generalised stability analyses <../documentation/gst>`.

- `IPOPT`_ and `pyipopt`_: This is the best available open-source optimisation algorithm. Strongly recommended if you wish to solve :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Make sure to compile IPOPT against the `Harwell Subroutine Library`_.

- `Moola`_: A set of optimisation algorithms specifically designed for :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Install with `pip install moola`. Note: still experimental.

.. _FEniCS: http://fenicsproject.org
.. _libadjoint: http://bitbucket.org/dolfin-adjoint/libadjoint
.. _SLEPc: http://www.grycap.upv.es/slepc/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _pyipopt: https://github.com/xuy/pyipopt
.. _moola: https://github.com/funsim/moola
.. _Harwell Subroutine Library: http://www.hsl.rl.ac.uk/ipopt/
.. _their installation instructions: http://fenicsproject.org/download
