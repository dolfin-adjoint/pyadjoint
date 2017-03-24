from .assembly import assemble
from .solving import solve
from .projection import project
from .functional import Functional
from .types import Function, Constant, DirichletBC, Expression
from pyadjoint.tape import Tape, set_working_tape, get_working_tape
from pyadjoint.reduced_functional import ReducedFunctional

tape = Tape()
set_working_tape(tape)
