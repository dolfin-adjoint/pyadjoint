import backend
from pyadjoint.adjfloat import AdjFloat
from .constant import Constant
from .dirichletbc import DirichletBC
if backend.__name__ != "firedrake":
    # Currently not implemented.
    from .expression import Expression, UserExpression, CompiledExpression
from .function import Function
from .function_space import FunctionSpace
from .mesh import Mesh
from .mesh import UnitSquareMesh
from .mesh import UnitIntervalMesh
from .mesh import IntervalMesh
from .mesh import UnitCubeMesh
from .mesh import RectangleMesh

from .types import create_overloaded_object
