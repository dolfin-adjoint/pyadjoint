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

# flake8: noqa

import pyadjoint
__version__ = pyadjoint.__version__
__author__ = 'Sebastian Kenji Mitusch'
__credits__ = []
__license__ = 'LGPL-3'
__maintainer__ = 'Sebastian Kenji Mitusch'
__email__ = 'sebastkm@math.uio.no'

import sys
if 'backend' not in sys.modules:
    import fenics
    sys.modules['backend'] = fenics
backend = sys.modules['backend']

from .assembly import assemble, assemble_system
from .solving import solve
from .projection import project
from .interpolation import interpolate
from .shapead_transformations import (transfer_from_boundary,
                                      transfer_to_boundary)

from dolfin_adjoint_common.ufl_constraints import UFLInequalityConstraint, UFLEqualityConstraint

if backend.__name__ != "firedrake":
    from .newton_solver import NewtonSolver
    from .lu_solver import LUSolver
    from .krylov_solver import KrylovSolver
    from .petsc_krylov_solver import PETScKrylovSolver
    from .types import *
    from .refine import refine
    from .system_assembly import *


from .variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                 LinearVariationalProblem, LinearVariationalSolver)
from pyadjoint import (Tape, set_working_tape, get_working_tape,
                       pause_annotation, continue_annotation,
                       ReducedFunctional,
                       taylor_test, taylor_to_dict,
                       compute_gradient, compute_hessian,
                       AdjFloat, Control, minimize, maximize, MinimizationProblem,
                       IPOPTSolver, ROLSolver, InequalityConstraint, EqualityConstraint,
                       MoolaOptimizationProblem, print_optimization_methods,
                       stop_annotating)


set_working_tape(Tape())
