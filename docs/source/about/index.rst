.. title:: dolfin-adjoint about

***************
About pyadjoint
***************

pyajoint is an operator-overloading algorithmic differentiation framework for
Python. It is employed as the basis for the automatic adjoint and tangent
linear model capabilities of the `FEniCS <http://dolfin-adjoint.org>`__ and 
`Firedrake <http://firedrakeproject.org>`__ projects.

These adjoint and tangent linear models are key ingredients in many
important algorithms, such as data assimilation, optimal control,
sensitivity analysis, design optimisation, and error estimation.  Such
models have made an enormous impact in fields such as meteorology and
oceanography, but their use in other scientific fields has been
hampered by the great practical difficulty of their derivation and
implementation. In `his book`_, Naumann (2011) states that

 [T]he automatic generation of optimal (in terms of robustness and
 efficiency) adjoint versions of large-scale simulation code is **one
 of the great open challenges in the field of High-Performance
 Scientific Computing**.

**The dolfin-adjoint project aims to solve this problem** for the case
where the model is implemented in the Python interface to FEniCS/Firedrake.

.. _his book: http://dx.doi.org/10.1137/1.9781611972078


.. _ChangeLog.rst: https://github.com/dolfin-adjoint/pyadjoint/blob/master/ChangeLog.rst
.. _available here: https://github.com/dolfin-adjoint/pyadjoint/blob/master/tests/migration/README.md
.. _contact us: support/index.html
.. _pyadjoint: https://github.com/dolfin-adjoint/pyadjoint
.. _documentation: http://dolfin-adjoint-doc.readthedocs.io/
.. _Wilkinson prize for numerical software: http://www.nag.co.uk/other/WilkinsonPrize.html
.. _poster: https://drive.google.com/file/d/1NjIFj07u_QMfuXB2Z8uv5f2LUDwY1XeM/view?usp=sharing




For more technical details on pyadjoint and dolfin-adjoint, :doc:`see
the papers <../citing/index>`.

Contributors
============

The dolfin-adjoint project is developed and maintained by the
following authors:

- `Sebastian Mitusch <https://www.simula.no/people/sebastkm>`__ (Simula Research Laboratory)
- `Jørgen S. Dokken <https://www.simula.no/people/dokken>`__ (Simula Research Laboratory)
- `Patrick E. Farrell <http://pefarrell.org>`__ (Mathematical Institute, University of Oxford)
- `Simon W. Funke <http://simonfunke.com>`__ (Simula Research Laboratory)
- `David A. Ham <http://www.ic.ac.uk/people/david.ham>`__ (Department of Mathematics and Department of Computing, Imperial College London)
- `Marie E. Rognes <http://home.simula.no/~meg/>`__ (Simula Research Laboratory)
- `James R. Maddison <http://www.maths.ed.ac.uk/people/show?person-364>`__ (School of Mathematics, University of Edinburgh)

Licence
=======

`pyadjoint` is
freely available under the `GNU LGPL
<http://www.gnu.org/licenses/lgpl.html>`__, version 3.
