.. _dolfin-adjoint-about:

.. title:: dolfin-adjoint about

*****
About
*****

The dolfin-adjoint project automatically **derives the discrete
adjoint and tangent linear models** from a forward model written in
the Python interface to `DOLFIN <http://fenicsproject.org>`__.

These adjoint and tangent linear models are key ingredients in many
important algorithms, such as data assimilation, optimal control,
sensitivity analysis, design optimisation, and error estimation.  Such
models have made an enormous impact in fields such as meteorology and
oceanography, but their use in other scientific fields has been
hampered by the great practical difficulty of their derivation and
implementation. In `his recent book`_, Naumann (2011) states that

 [T]he automatic generation of optimal (in terms of robustness and
 efficiency) adjoint versions of large-scale simulation code is **one
 of the great open challenges in the field of High-Performance
 Scientific Computing**.

**The dolfin-adjoint project aims to solve this problem** for the case
where the model is implemented in the Python interface to DOLFIN.

.. _his recent book: http://dx.doi.org/10.1137/1.9781611972078


News
====
**31.05.2017**: dolfin-adjoint 2017.1 released (compatible with FEniCS 2017.1) `ChangeLog`_.

**21.02.2017**: dolfin-adjoint 2016.2 is now available as Conda package.

**20.12.2016**: dolfin-adjoint 2016.2 released (compatible with FEniCS 2016.2) `Changelog`_.

**08.9.2016**: We now offer Docker images to simplify the installation procedure.

**30.8.2016**: dolfin-adjoint 2016.1 released (compatible with FEniCS 2016.1) `Changelog`_.

**20.8.2015**: dolfin-adjoint 1.6 released (compatible with FEniCS 1.6) `Changelog`_.

**11.6.2015**: P. E. Farrell, S. W. Funke, D. A. Ham and M. E. Rognes were awarded the 2015 `Wilkinson prize for numerical software`_ for dolfin-adjoint.

.. _Changelog: https://bitbucket.org/dolfin-adjoint/dolfin-adjoint/raw/master/ChangeLog
.. _Wilkinson prize for numerical software: http://www.nag.co.uk/other/WilkinsonPrize.html

Features
========

dolfin-adjoint has the following features:

- Works for both steady and **time-dependent problems** and for both linear and **nonlinear problems**.
- Using it is very **easy**: given a differentiable forward model, employing dolfin-adjoint involves
  changing on the order of ten lines of code.
- The adjoint and tangent linear models exhibit **optimal theoretical efficiency**. If every forward
  variable is stored, the adjoint takes 0.2-1.0x the runtime of the forward model, depending on the
  precise details of the structure of the forward problem.
- If the forward model runs in parallel, the adjoint and tangent linear models also **run in parallel**
  with no modification.
- If instructed, the adjoint model can automatically employ **optimal checkpointing** schemes to
  mitigate storage requirements for long nonlinear runs.
- **Rigorous verification** routines are provided, so that users can easily verify for themselves
  the correctness of the derived models.
- Solves **optimisation problems** constrained by partial differential equations by interfacing to powerful optimisation algorithms

For more details, :doc:`see the features page<../features/index>`.

Limitations
===========

To do all this, dolfin-adjoint requires some cooperation from the
model developer:

- Works only with the Python interface of DOLFIN.
- For the adjoint to be consistent, the discretisation must be differentiable.
- All changes to object values (matrices, vectors, functions) must happen through the DOLFIN interface.


How it works
============

The traditional approach to deriving adjoint and tangent linear models
is called `algorithmic differentiation`_ (also called automatic
differentiation). The fundamental idea of algorithmic differentiation
is to *treat the model as a sequence of elementary instructions*. An
elementary instruction is a simple operation such as addition,
multiplication, or exponentiation. Each one of these operations is
differentiated individually, and the derivative of the whole model is
then composed with the chain rule.

.. _algorithmic differentiation: http://www.autodiff.org

The dolfin-adjoint project is instead based on a very different
approach.  The model is considered as *a sequence of equation
solves*. This abstraction is similar to the fundamental abstraction of
algorithmic differentiation, but operates at a much higher level of
abstraction. This idea is implemented in a software library,
`libadjoint`_. When this new idea is combined with the high-level
abstraction of the FEniCS system, many of the difficult problems
associated with algorithmic differentiation dissolve.

For more technical details on libadjoint and dolfin-adjoint, :doc:`see
the papers <../citing/index>`.

.. _libadjoint: http://bitbucket.org/dolfin-adjoint/libadjoint

Contributors
============

The dolfin-adjoint project is developed and maintained by the
following authors:

- `Patrick E. Farrell <http://pefarrell.org>`__ (Mathematical Institute, University of Oxford)
- `Simon W. Funke <http://simonfunke.com>`__ (Center for Biomedical Computing, Simula Research Laboratory)
- `David A. Ham <http://www.ic.ac.uk/people/david.ham>`__ (Department of Mathematics and Department of Computing, Imperial College London)
- `Marie E. Rognes <http://home.simula.no/~meg/>`__ (Center for Biomedical Computing, Simula Research Laboratory)
- `James R. Maddison <http://www.maths.ed.ac.uk/people/show?person-364>`__ (School of Mathematics, University of Edinburgh)

License
=======

Like the `core FEniCS components`_, The dolfin-adjoint software is
freely available under the `GNU LGPL
<http://www.gnu.org/licenses/lgpl.html>`__, version 3.

.. _core FEniCS components: http://fenicsproject.org/about/
