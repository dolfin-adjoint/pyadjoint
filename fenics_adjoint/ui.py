import backend
from .assembly import assemble, assemble_system
from .solving import solve
from .projection import project
from .types import Function, Constant, DirichletBC, FunctionSpace
if backend.__name__ != "firedrake":
    from .types import Expression
from .variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                 LinearVariationalProblem, LinearVariationalSolver)
from pyadjoint import (Tape, set_working_tape, get_working_tape,
                       pause_annotation, continue_annotation,
                       ReducedFunctional,
                       taylor_test, taylor_test_multiple,
                       compute_gradient, Hessian,
                       AdjFloat)

tape = Tape()
set_working_tape(tape)
