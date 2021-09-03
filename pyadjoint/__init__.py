# flake8: noqa

__version__ = '2019.1.2'
__author__  = 'Sebastian Kenji Mitusch'
__credits__ = []
__license__ = 'LGPL-3'
__maintainer__ = 'Sebastian Kenji Mitusch'
__email__ = 'sebastkm@math.uio.no'

from .block import Block
from .tape import (Tape,
                   set_working_tape, get_working_tape, no_annotations,
                   annotate_tape, stop_annotating, pause_annotation, continue_annotation)
from .adjfloat import AdjFloat
from .reduced_functional import ReducedFunctional
from .drivers import compute_gradient, compute_hessian, solve_adjoint
from .verification import taylor_test, taylor_to_dict
from .overloaded_type import OverloadedType, create_overloaded_object
from .control import Control
from .optimization.optimization import minimize, maximize, print_optimization_methods
from .optimization.optimization_problem import MinimizationProblem
from .optimization.ipopt_solver import IPOPTSolver
from .optimization.rol_solver import ROLSolver
from .optimization.constraints import InequalityConstraint, EqualityConstraint
from .optimization.moola_problem import MoolaOptimizationProblem
