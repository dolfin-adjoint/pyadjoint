:orphan:

.. features

.. title:: dolfin-adjoint features

.. py:currentmodule:: fenics_adjoint

**************************
Features of dolfin-adjoint
**************************

Generality
==========

dolfin-adjoint works for both steady and time-dependent models, and for both linear
and nonlinear models. The user interface is exactly the same in both cases. For an example
of adjoining a nonlinear time-dependent model, see the :doc:`tutorial <../documentation/tutorial>`.

Ease of use
===========

dolfin-adjoint has been carefully designed to try to make its use as easy as possible. In many cases
the only change to the forward model is to add

.. code-block:: python

    from dolfin_adjoint import *

at the top of the model. For example, deriving the adjoint of :doc:`the tutorial example <../documentation/tutorial>` requires **adding
precisely three lines to the forward model**. dolfin-adjoint also makes it extremely easy to :doc:`verify the correctness of the adjoint model <../documentation/verification>`.
It offers a powerful syntax for :doc:`expressing general functionals <../documentation/functionals>`.

Efficiency
==========

Efficiency of the resulting model is absolutely crucial for real applications. The efficiency of
an adjoint model is measured as (time for forward and adjoint run)/(time for forward run). `Naumann (2011)`_ states
that a typical value for this ratio when using algorithmic differentiation tools is in the range 3--30. By contrast, dolfin-adjoint is **extremely efficient**;
consider the following examples from :doc:`the papers <../citing/index>`:

.. _Naumann (2011): http://dx.doi.org/10.1137/1.9781611972078

.. tabularcolumns:: |c|c|c|

.. list-table::
    :widths: 15, 10, 15
    :header-rows: 1
    :class: center

    * - PDE

      - Theoretical optimum

      - Achieved efficiency

    * - Cahn-Hilliard

      - 1.2

      - 1.22

    * - Stokes

      - 2.0

      - 1.86

    * - Viscoelasticity

      - 2.0

      - 2.029

    * - Gross-Pitaevskii

      - 1.5

      - 1.54

    * - Gray-Scott

      - 2.0

      - 2.04

    * - Navier-Stokes

      - 1.33

      - 1.41

    * - Mathematical programming with equilibrium constraints

      - 1.125

      - 1.126

    * - Shallow water

      - 1.125

      - 1.125

    * - Wetting and drying

      - 1.5

      - 1.55

Parallelism
===========

Parallelism is ubiquitous in modern computational science. However,
applying algorithmic differentiation to parallel codes is still a
major research challenge. Algorithmic differentiation tools must be
specially modified to understand MPI and OpenMP directives, and
translate them into their parallel equivalents. By contrast, *because
of the high-level abstraction taken in libadjoint, the problem of
parallelism simply disappears*. In fact, there is no code whatsoever
in either dolfin-adjoint or pyadjoint to handle parallelism; by
deriving the adjoint at the right level of abstraction, the problem no
longer exists.  **If the forward model runs in parallel, the adjoint
model also runs in parallel, with no modification.**

For more details, see :doc:`the manual section on parallelism
<../documentation/parallel>` and :doc:`the dolfin-adjoint paper
<../citing/index>`.
