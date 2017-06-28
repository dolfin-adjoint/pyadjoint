.. features

.. title:: dolfin-adjoint features

.. py:currentmodule:: dolfin_adjoint

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
precisely two lines to the forward model**; :doc:`implementing a checkpointing scheme <../documentation/checkpointing>` requires adding
another two. dolfin-adjoint also makes it extremely easy to :doc:`verify the correctness of the adjoint model <../documentation/verification>`.
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
in either dolfin-adjoint or libadjoint to handle parallelism; by
deriving the adjoint at the right level of abstraction, the problem no
longer exists.  **If the forward model runs in parallel, the adjoint
model also runs in parallel, with no modification.**

For more details, see :doc:`the manual section on parallelism
<../documentation/parallel>` and :doc:`the dolfin-adjoint paper
<../citing/index>`.

Checkpointing
=============

The adjoint model is a *linearisation* of the forward model. If the
forward model is nonlinear, then the solution of that forward model
must be available to *linearise* the forward model. By default,
dolfin-adjoint stores every variable computed in memory, as this is
the fastest and most straightforward option; however, this may not be
feasible for large runs, or for runs with very many timesteps.

The solution to this problem is to employ a *checkpointing
scheme*. Rather than store every variable during the forward run,
checkpoints are stored at strategically chosen intervals, from which
the model may recompute the missing solutions. During the adjoint run,
if a forward variable is necessary and unavailable, the forward model
is restarted from the nearest available checkpoint to compute the
missing solutions; once these are available, the adjoint run
continues.

Thus, to employ a checkpointing scheme, the control flow of the
adjoint run must seamlessly jump between assembling and solving the
adjoint equations, and assembling and solving parts of the forward
run. Coding a checkpointing scheme is quite complicated, and so most
hand-coded adjoint models do not use them. However, the
:doc:`libadjoint library underlying dolfin-adjoint <../citing/index>`
embeds the excellent `revolve library`_ of `Griewank and Walther`_,
and can **automatically employ optimal checkpointing schemes for
almost no marginal user effort**.

.. _revolve library: http://www2.math.uni-paderborn.de/index.php?id=12067&L=1
.. _Griewank and Walther: http://dx.doi.org/10.1145/347837.347846

For more details, see :doc:`the manual section on checkpointing <../documentation/checkpointing>` and :doc:`the dolfin-adjoint paper <../citing/index>`.


Optimisation with PDE constraints
=================================

Many computational problems in engineering and science can be
formulated as optimisation problems in which a system of partial
differential equation occur as a constraint.  To solve these problems
efficiently, the use of gradient based optimisation algorithms is
essential.

The fact that dolfin-adjoint has direct access to the gradient
information made it possible to directly interface dolfin-adjoint to a
range of powerful optimisation algorithms.  That means, that **an
existing FEniCS forward model can be easily used in the context of an
optimisation problem** with minimal development effort.

For more details, see :doc:`the manual section on optimisation
<../documentation/optimisation>`.

Generalised stability analysis
==============================

Generalised stability analysis is an extension of linear stability analysis with
two important features: it allows for the stability analysis of non-normal systems
that permit transient perturbations that grow in magnitude before decaying, and
it allows for the analysis of the stability of time-dependent base solutions.

The core computation involved in conducting a generalised stability analysis is
the singular value decomposition of the *propagator*: each action of the
propagator requires the solution of the tangent linear system, and each adjoint
action (for the singular value decomposition) requires the solution of the
adjoint system. Since dolfin-adjoint automates the solution of these systems, it
can also automate generalised stability analysis by embedding these computations
in a matrix-free Krylov--Schur algorithm for computing the SVD of the
propagator.

For more details, see :doc:`the manual section on generalised stability analysis
<../documentation/gst>`.

