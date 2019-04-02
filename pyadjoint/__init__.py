__version__ = '2018.1.0.r1'
__author__  = 'Sebastian Kenji Mitusch'
__credits__ = []
__license__ = 'LGPL-3'
__maintainer__ = 'Sebastian Kenji Mitusch'
__email__ = 'sebastkm@math.uio.no'

from .block import Block
from .tape import (Tape,
                   set_working_tape, get_working_tape,
                   annotate_tape, stop_annotating, pause_annotation, continue_annotation)
from .adjfloat import AdjFloat
from .reduced_functional import ReducedFunctional
from .drivers import compute_gradient, compute_hessian
from .verification import taylor_test
from .overloaded_type import OverloadedType
from .control import Control
from .optimization.optimization import minimize, print_optimization_methods
from .optimization.optimization_problem import MinimizationProblem
from .optimization.ipopt_solver import IPOPTSolver
from .optimization.rol_solver import ROLSolver
from .optimization.constraints import InequalityConstraint, EqualityConstraint
from .optimization.moola_problem import MoolaOptimizationProblem
