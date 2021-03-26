# flake8: noqa

import pyadjoint
__version__ = pyadjoint.__version__

import sys
if 'backend' not in sys.modules:
    import firedrake
    sys.modules['backend'] = firedrake
else:
    raise ImportError("'backend' module already exists?")

from pyadjoint.tape import (Tape, set_working_tape, get_working_tape,
                            pause_annotation, continue_annotation,
                            stop_annotating, annotate_tape)
from pyadjoint.reduced_functional import ReducedFunctional
from pyadjoint.verification import taylor_test, taylor_to_dict
from pyadjoint.drivers import compute_gradient, compute_hessian
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.control import Control
from pyadjoint import IPOPTSolver, ROLSolver, MinimizationProblem, InequalityConstraint, minimize

from dolfin_adjoint_common.ufl_constraints import UFLInequalityConstraint, UFLEqualityConstraint
import numpy_adjoint

continue_annotation()
set_working_tape(Tape())
