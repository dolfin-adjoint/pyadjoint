import backend
from .assembly import assemble, assemble_system
from .solving import solve
from .projection import project
from .types import Function, Constant, DirichletBC, FunctionSpace
if backend.__name__ != "firedrake":
    from .types import Expression
from .variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                 LinearVariationalProblem, LinearVariationalSolver)
from pyadjoint.tape import Tape, set_working_tape, get_working_tape
from pyadjoint.reduced_functional import ReducedFunctional
from pyadjoint.verification import taylor_test, taylor_test_multiple
from pyadjoint.drivers import compute_gradient, Hessian
from pyadjoint.adjfloat import AdjFloat

tape = Tape()
set_working_tape(tape)
