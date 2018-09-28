import backend
from pyadjoint.adjfloat import AdjFloat
from .constant import Constant
from .dirichletbc import DirichletBC
if backend.__name__ != "firedrake":
    # Currently not implemented.
    from .expression import Expression
from .function import Function

from .types import create_overloaded_object
