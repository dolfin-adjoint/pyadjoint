.. _dolfin-adjoint-difference:

.. title:: dolfin-adjoint Difference between dolfin-adjoint/pyadjoint and dolfin-adjoint/libadjoint

**************************************************************************************
What is the difference between dolfin-adjoint/pyadjoint and dolfin-adjoint/libadjoint?
**************************************************************************************

dolfin-adjoint/libadjoint
*************************
This is the original implementation of dolfin-adjoint and uses the C library `libadjoint`_ 
as the underlying differentiation tool.

dolfin-adjoint/libadjoint is not longer under development. The last release was 
version 2017.2.0. The documentation for dolfin-adjoint/libadjoint can still be accessed `here`_.

Existing software projects that already use dolfin-adjoint/libadjoint are encouraged 
to eventually update their code to dolfin-adjoint/pyadjoint. Since both implementations
are mostly API compatible, the required changes should be relatively small.


dolfin-adjoint/pyadjoint 
*************************

This is a full rewrite of dolfin-adjoint based on the Python algorithmic differentiation tool `pyadjoint`_ (see this `poster`_).

This version will be maintained and developed in the future. New projects are advised to use this version instead of dolfin-adjoint/libadjoint.

Compared to dolfin-adjoint/libadjoint, dolfin-adjoint/pyadjoint is superiour 
in some features, for instance it has full Hessian support, a more generic  
way of defining functionals and support for Dirichlet BC controls. 


A list of dolfin-adjoint/libadjoint tests that  `available here`_. 
If you would like to contribute, please `contact us`_.
                
.. _available here: https://bitbucket.org/dolfin-adjoint/pyadjoint/src/master/tests/migration/README.md?at=master&fileviewer=file-view-default
.. _contact us: support/index.html
.. _pyadjoint: https://bitbucket.org/dolfin-adjoint/pyadjoint
.. _libadjoint: https://bitbucket.org/dolfin-adjoint/libadjoint
.. _here: http://dolfin-adjoint-doc.readthedocs.io/
.. _poster: https://drive.google.com/file/d/1NjIFj07u_QMfuXB2Z8uv5f2LUDwY1XeM/view?usp=sharing
