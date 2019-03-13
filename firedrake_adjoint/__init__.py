# flake8: noqa

import pyadjoint
__version__ = pyadjoint.__version__

import sys
if 'backend' not in sys.modules:
    import firedrake
    sys.modules['backend'] = firedrake
else:
    raise ImportError("'backend' module already exists?")

# flake8: noqa F401
from fenics_adjoint.assembly import assemble, assemble_system
from fenics_adjoint.solving import solve
from fenics_adjoint.projection import project
from fenics_adjoint.types import Constant, DirichletBC
from fenics_adjoint.variational_solver import (NonlinearVariationalProblem, NonlinearVariationalSolver,
                                               LinearVariationalProblem, LinearVariationalSolver)
from fenics_adjoint.interpolation import interpolate
from fenics_adjoint.ufl_constraints import UFLInequalityConstraint, UFLEqualityConstraint

from firedrake_adjoint.types.expression import Expression
from firedrake_adjoint.types.function import Function

from pyadjoint.tape import (Tape, set_working_tape, get_working_tape,
                            pause_annotation, continue_annotation)
from pyadjoint.reduced_functional import ReducedFunctional
from pyadjoint.verification import taylor_test
from pyadjoint.drivers import compute_gradient, compute_hessian
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.control import Control
from pyadjoint import IPOPTSolver, ROLSolver, MinimizationProblem, InequalityConstraint, minimize

import firedrake
import sys
thismod = sys.modules[__name__]
meshes = __import__("firedrake_adjoint.types.mesh",
                    fromlist=firedrake.utility_meshes.__all__)
for name in firedrake.utility_meshes.__all__:
    setattr(thismod, name, getattr(meshes, name))
from firedrake_adjoint.types.mesh import Mesh


set_working_tape(Tape())
