# flake8: noqa

import pyadjoint
__version__ = pyadjoint.__version__

import sys
if 'backend' not in sys.modules:
    import firedrake
    sys.modules['backend'] = firedrake
else:
    raise ImportError("'backend' module already exists?")

# flake8: noqa F401
from fenics_adjoint.assembly import assemble, assemble_system
from fenics_adjoint.solving import solve
from fenics_adjoint.projection import project
from fenics_adjoint.types import Constant, DirichletBC
from fenics_adjoint.variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                               LinearVariationalProblem, LinearVariationalSolver)
from fenics_adjoint.interpolation import interpolate
from fenics_adjoint.ufl_constraints import UFLInequalityConstraint, UFLEqualityConstraint

from pyadjoint.tape import (Tape, set_working_tape, get_working_tape,
                            pause_annotation, continue_annotation,
                            stop_annotating)
from pyadjoint.reduced_functional import ReducedFunctional
from pyadjoint.verification import taylor_test, taylor_to_dict
from pyadjoint.drivers import compute_gradient, compute_hessian
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.control import Control
from pyadjoint import IPOPTSolver, ROLSolver, MinimizationProblem, InequalityConstraint, minimize

from firedrake_adjoint.types import *


set_working_tape(Tape())
