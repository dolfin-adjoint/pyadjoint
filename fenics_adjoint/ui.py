from .assembly import assemble
from .solving import solve
from .projection import project
from .types import Function, Constant, DirichletBC, Expression
from pyadjoint.tape import Tape, set_working_tape, get_working_tape
from pyadjoint.reduced_functional import ReducedFunctional
from pyadjoint.verification import taylor_test

tape = Tape()
set_working_tape(tape)
