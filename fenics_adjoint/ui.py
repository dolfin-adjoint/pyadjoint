import backend
from .assembly import assemble, assemble_system
from .solving import solve
from .projection import project
from .interpolation import interpolate
from .types import Function, Constant, DirichletBC, FunctionSpace, Mesh
if backend.__name__ != "firedrake":
    from .types import Expression, UserExpression
    from .types import io
    from .newton_solver import NewtonSolver
    from .lusolver import LUSolver
from .variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                 LinearVariationalProblem, LinearVariationalSolver)
from pyadjoint import (Tape, set_working_tape, get_working_tape,
                       pause_annotation, continue_annotation,
                       ReducedFunctional,
                       taylor_test, taylor_test_multiple,
                       compute_gradient, compute_hessian,
                       AdjFloat, Control, minimize, MinimizationProblem,
                       IPOPTSolver, InequalityConstraint)

tape = Tape()
set_working_tape(tape)
