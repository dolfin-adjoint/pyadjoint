# flake8: noqa

import backend
import sys
from pyadjoint.adjfloat import AdjFloat
from .constant import Constant
from .dirichletbc import DirichletBC
if backend.__name__ != "firedrake":
    # Currently not implemented.
    from .expression import Expression, UserExpression, CompiledExpression

    # Shape AD specific imports for dolfin
    thismod = sys.modules[__name__]
    # FIXME: We do not support UnitDiscMesh, SphericalShellMesh
    # and UnitTriangleMesh, as the do not have an initializer, only "def create"
    overloaded_meshes = ['IntervalMesh', 'UnitIntervalMesh', 'RectangleMesh',
                         'UnitSquareMesh', 'UnitCubeMesh', 'BoxMesh']

    meshes = __import__("types.mesh", level=1, globals={"__name__": __name__},
                        fromlist=overloaded_meshes)
    for name in overloaded_meshes:
        setattr(thismod, name, getattr(meshes, name))
    from .mesh import (Mesh, SubMesh, BoundaryMesh)

    from .as_backend_type import as_backend_type, VectorSpaceBasis
from .function import Function

# Use pyadjoint AdjFloat for numpy.float64.
import numpy
from pyadjoint.overloaded_type import register_overloaded_type
from pyadjoint.adjfloat import AdjFloat
register_overloaded_type(AdjFloat, numpy.float64)

