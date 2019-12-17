# flake8: noqa F401
from pyadjoint.tape import (Tape, set_working_tape, get_working_tape,
                            pause_annotation, continue_annotation)
from pyadjoint.reduced_functional import ReducedFunctional
from pyadjoint.verification import taylor_test
from pyadjoint.drivers import compute_gradient, compute_hessian
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.control import Control
from pyadjoint import IPOPTSolver, ROLSolver, MinimizationProblem, InequalityConstraint, minimize

tape = Tape()
set_working_tape(tape)
