.. _download:

*************************
Installing dolfin-adjoint
*************************

**Note**: If you are looking to install the (deprecated) dolfin-adjoint/libadjoint library, visit the `dolfin-adjoint/libadjoint`_ webpage.

:ref:`dolfin-adjoint-difference`

.. _dolfin-adjoint/libadjoint: http://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html

Source code
===========

The source code of `pyadjoint` is available on https://bitbucket.org/dolfin-adjoint/pyadjoint.

PIP (all platforms, recommended)
================================

Install dolfin-adjoint and its Python dependencies with pip:

.. code-block:: bash

    pip install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@2018.1.0

Test your installation by running:

.. code-block:: bash

    python3 -c "import fenics_adjoint"

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




Docker images (all platforms)
=============================

`Docker <https://www.docker.com>`_ allows us to build and ship
consistent high-performance dolfin-adjoint installations with FEniCS for almost any
platform. To get started, follow these 2 steps:

#. Install Docker. Mac and Windows users should install the `Docker
   Toolbox <https://www.docker.com/products/docker-toolbox>`_ (this is
   a simple one-click install) and Linux users should `follow these
   instructions <https://docs.docker.com/linux/step_one/>`_.
#. Install the FEniCS Docker script::

    curl -s https://get.fenicsproject.org | bash

If running on Mac or Windows, make sure you run this and other
commands inside the Docker Quickstart Terminal.

Stable version:
---------------
Once both Docker and the FEniCS Docker script have been installed, you can
easily start a FEniCS session with dolfin-adjoint by running the following
command::

    fenicsproject run quay.io/dolfinadjoint/pyadjoint

A Jupyter notebook instance with a user defined name (here myproject) can be started with::

    fenicsproject notebook myproject quay.io/dolfinadjoint/pyadjoint
    fenicsproject start myproject

The FEniCS Docker script can also be used to create persistent sessions::

    fenicsproject create myproject quay.io/dolfinadjoint/pyadjoint
    fenicsproject start myproject

To create a session that has access to the current folder from the host::

    docker run -ti -v $(pwd):/home/fenics/shared quay.io/dolfinadjoint/pyadjoint

Development version:
--------------------
The development version of dolfin-adjoint and FEniCS is available with::

    fenicsproject run quay.io/dolfinadjoint/dev-pyadjoint


To update the development container, run::

    fenicsproject pull quay.io/dolfinadjoint/dev-pyadjoint

To see all the options run::

    fenicsproject help

For more details and tips on how to work with FEniCS and Docker, see
our `FEniCS Docker page
<http://fenics-containers.readthedocs.org/en/latest/>`_.

