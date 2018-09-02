import backend
from .assembly import assemble, assemble_system
from .refine import refine
from .solving import solve
from .projection import project
from .interpolation import interpolate
from .types import (Function, Constant, DirichletBC, FunctionSpace,
                    Mesh, UnitSquareMesh, UnitIntervalMesh, IntervalMesh,
                    UnitCubeMesh, RectangleMesh)
if backend.__name__ != "firedrake":
    from .types import Expression, UserExpression, CompiledExpression
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
                       IPOPTSolver, InequalityConstraint,
                       MoolaOptimizationProblem)

tape = Tape()
set_working_tape(tape)
