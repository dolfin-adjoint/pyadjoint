"""

The entire dolfin-adjoint interface should be imported with a single
call:

.. code-block:: python

  from dolfin import *
  from dolfin_adjoint import *

It is essential that the importing of the :py:mod:`dolfin_adjoint` module happen *after*
importing the :py:mod:`dolfin` module. dolfin-adjoint relies on *overloading* many of
the key functions of dolfin to achieve its degree of automation.
"""

import pyadjoint
__version__ = pyadjoint.__version__
__author__ = 'Sebastian Kenji Mitusch'
__credits__ = []
__license__ = 'LGPL-3'
__maintainer__ = 'Sebastian Kenji Mitusch'
__email__ = 'sebastkm@math.uio.no'

import sys
if not 'backend' in sys.modules:
    import fenics
    sys.modules['backend'] = fenics
backend = sys.modules['backend']

if backend.__name__ != "firedrake":
    from .types import genericmatrix
    from .types import genericvector

from .ui import *
