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
from firedrake.adjoint.checkpointing import (
    enable_disk_checkpointing, pause_disk_checkpointing,
    continue_disk_checkpointing, stop_disk_checkpointing,
    checkpointable_mesh
)
from pyadjoint.verification import taylor_test, taylor_to_dict
from pyadjoint.drivers import compute_gradient, compute_hessian
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.control import Control
from pyadjoint import IPOPTSolver, ROLSolver, MinimizationProblem, InequalityConstraint, minimize

from firedrake.adjoint.ufl_constraints import UFLInequalityConstraint, UFLEqualityConstraint
import numpy_adjoint

continue_annotation()
set_working_tape(Tape())
