# flake8: noqa

import backend

from .constant import Constant
from .dirichletbc import DirichletBC
if backend.__name__ != "firedrake":
    # Currently not implemented.
    from .expression import Expression, UserExpression, CompiledExpression

    # Shape AD specific imports for dolfin
    from .mesh import *

    from .genericmatrix import *
    from .genericvector import *
    from .io import *

    from .as_backend_type import as_backend_type, VectorSpaceBasis
    from .function_assigner import *
from .function import Function

# Import numpy_adjoint to annotate numpy outputs
import numpy_adjoint
