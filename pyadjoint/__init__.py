# flake8: noqa
from .optimization.moola_problem import MoolaOptimizationProblem
from .optimization.constraints import InequalityConstraint, EqualityConstraint
from .optimization.tao_solver import TAOSolver
from .optimization.rol_solver import ROLSolver
from .optimization.ipopt_solver import IPOPTSolver
from .optimization.optimization_problem import MinimizationProblem
from .optimization.optimization import (
    SciPyConvergenceError,
    minimize,
    maximize,
    print_optimization_methods,
)
from .control import Control
from .overloaded_type import OverloadedType, create_overloaded_object
from .verification import taylor_test, taylor_to_dict
from .drivers import compute_gradient, compute_derivative, compute_tlm, compute_hessian, solve_adjoint
from .checkpointing import disk_checkpointing_callback
from .reduced_functional import ReducedFunctional
from .adjfloat import AdjFloat, exp, log
from .tape import (
    Tape,
    set_working_tape,
    get_working_tape,
    no_annotations,
    annotate_tape,
    stop_annotating,
    pause_annotation,
    continue_annotation,
)
from .block import Block
from importlib.metadata import metadata

meta = metadata("pyadjoint-ad")
__version__ = meta["Version"]
__author__ = meta.get("Author", "")
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


__all__ = [
    "Block",
    "Tape",
    "set_working_tape",
    "get_working_tape",
    "no_annotations",
    "annotate_tape",
    "stop_annotating",
    "pause_annotation",
    "continue_annotation",
    "AdjFloat",
    "exp",
    "log",
    "Control",
    "ReducedFunctional",
    "create_overloaded_object",
    "OverloadedType",
    "compute_gradient",
    "compute_derivative",
    "compute_tlm",
    "compute_hessian",
    "solve_adjoint",
    "taylor_test",
    "taylor_to_dict",
    "disk_checkpointing_callback",
    "MoolaOptimizationProblem",
    "InequalityConstraint",
    "EqualityConstraint",
    "TAOSolver",
    "ROLSolver",
    "IPOPTSolver",
    "MinimizationProblem",
    "SciPyConvergenceError",
    "minimize",
    "maximize",
    "print_optimization_methods",
]
