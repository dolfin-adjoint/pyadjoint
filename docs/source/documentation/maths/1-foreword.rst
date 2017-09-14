========
Foreword
========

*Written by Patrick E. Farrell*

************************
Why care about adjoints?
************************

Far too often, maths books launch into their subject without
explaining to the novice reader why he or she should care about it in
the first place. So, before diving into the details, let's take a few
minutes to motivate why adjoint techniques were invented.

Suppose an aeronautical engineer wishes to design a wing. The wing is
parametrised by a vector :math:`m`; for example, suppose each entry of
:math:`m` is the coefficient of a Bézier curve. For any potential wing
design :math:`m`, the Euler equations can be solved, and the
lift-to-drag ratio :math:`J` of the design computed.  With an adjoint,
the engineer can do far more: the adjoint computes *the derivative of
the drag with respect to the design parameters*. This can be used to
guide a human designer, or can be passed to an automated optimisation
algorithm to automatically compute an optimal
shape. :cite:`jameson1988` :cite:`giles2000`.  In the literature, this
concept is referred to as adjoint design optimisation.

Suppose a meteorologist wishes to improve a forecast by constraining
the weather model to match atmospheric observations. The state of the
atmosphere at the initial time is partially known (from weather
stations), but in order to initialise the model an initial condition
for the whole world is required. For any guess of the (unknown)
initial state of the atmosphere :math:`m`, the Navier-Stokes and
related equations can be solved, and the weighted misfit :math:`J`
between the observed values and the simulation results can be
computed. With an adjoint, the meteorologist can *systematically
update their guess for the initial state of the atmosphere* to match
the observations :cite:`ledimet1986` :cite:`talagrand1987`. In the
literature, this concept is referred to as variational data
assimilation, 3D-Var and 4D-Var.

Suppose an oceanographer wishes to understand the impact of bottom
topography on transport through the Drake passage. Bottom topography
(the shape of the sea floor) is quite poorly known; many areas of the
world are sparsely observed, and observations from over a century ago
are still used in some places. The bottom topography is represented as
a scalar field :math:`m`, the Navier-Stokes and related equations are
solved, and the average net transport through the Drake passage
:math:`J` computed. With an adjoint, the oceanographer can see *where
the transport is most sensitive to the topography*, and so quantify
where the uncertainty matters most :cite:`losch2007`. In the
literature, this concept is referred to as sensitivity analysis.

Suppose a nuclear engineer working for a government regulator wishes
to examine a proposed new nuclear reactor design. To do this, a
forward model of the Boltzmann transport equations will be used to
simulate the proposed design and verify its safety. However, all
simulations inherently come with discretisation errors, and unless
those errors are quantified, the simulations cannot be relied upon to
make decisions upon which millions of lives and billions of pounds
depend. With an adjoint, the engineer can *quantify the impact of
discretisation errors* on the criticality rate, and decide to what
extent the simulations may be trusted :cite:`becker2001`. In the
literature, this concept is referred to as goal-based error
estimation, or goal-based adaptivity.

Suppose a mathematician wishes to understand the stability of some
physical system. The traditional approach to this problem is to
linearise the operator and investigate its eigenvalues, which
determine the long-term behaviour of the system (as :math:`t
\rightarrow \infty`). However, systems that are eigenvalue-stable can
exhibit unexpected transient growth of small perturbations, which in
turn can cause the system to become unstable (through nonlinear
effects) :cite:`trefethen1993`. By computing the singular value
decomposition of the tangent linear model, *the transient growth of
the system to such perturbations can be quantified, and the optimally
growing perturbations identified* :cite:`farrell1996`.  The
computation of the singular value decomposition in turn requires the
adjoint. In the literature, this approach is referred to as
generalised stability theory.

As you can see, adjoints show up in many applications, and in many
computational techniques.  One of the reasons why adjoints have a
reputation for being difficult is because their discussion is
performed in many different areas of science, usually with their own
specialised terminology.  Reading the literature, there are almost as
many ways to approach the topic as there are practitioners!  With this
introduction, I hope to strike to the heart of the matter, and clear
some of the confusion with the minimum of application-- or
technique--specific lingo.

************************
A note on the exposition
************************

I have chosen to motivate adjoints via a discussion of
*PDE-constrained optimisation* for two reasons. The first is that this
approach encapsulates many important applications of adjoints in a
general way, and so the reader will be well-equipped to understand
much adjoint-related mathematics in the literature.  The second is the
elegance of the result: most people are amazed when they first learn
that it is possible to compute the gradient of a functional
:math:`\widehat{J}(m)` in a cost *independent of the number of
parameters* :math:`\textrm{dim}(m)`! The topic of adjoints is
intriguing, counterintuitive and beautiful; any exposition should try
to live up to that.

The focus of the exposition will be on getting the core ideas across,
and for this reason the discussion will sometimes neglect
technicalities. For example, I will implicitly assume that all
problems are well-posed, that all necessary derivatives exist and are
sufficiently smooth, etc. Occasionally, to build intuition, I will
refer to objects as matrices and vectors, although the exposition
holds in exactly the same way for their analogues in functional
analysis.  For an advanced in-depth technical treatment of
PDE-constrained optimisation, see the excellent book of Hinze et
al. :cite:`hinze2009`.

********
Notation
********

The notation is mostly inspired by Gunzburger :cite:`gunzburger2003`.

.. tabularcolumns:: |c|c|

.. list-table::
    :widths: 15, 15
    :header-rows: 1
    :class: center

    * - Symbol

      - Meaning

    * - :math:`m`

      - the vector of parameters

    * - :math:`u`

      - the solution of the PDE

    * - :math:`F(u, m)`

      - the PDE relating :math:`u` and :math:`m`: :math:`F \equiv 0`

    * - :math:`J(u, m)`

      - a functional of interest

    * - :math:`\widehat{J}(m)`

      - the functional considered as a pure function of :math:`m`: :math:`\widehat{J}(m) = J(u(m), m)`

In :doc:`the next section <2-problem>`, we introduce the
PDE-constrained optimisation problem and give a broad overview of how
it may be tackled.

.. rubric:: References

.. bibliography:: 1-foreword.bib
   :cited:
   :labelprefix: 1M-
