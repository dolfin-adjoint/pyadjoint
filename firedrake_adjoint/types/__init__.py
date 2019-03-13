import firedrake
import sys

from .function import Function
from .mesh import Mesh

__all__ = ["Function", "Mesh"]


thismod = sys.modules[__name__]
meshes = __import__("types.mesh", level=1, globals={"__name__": __name__},
                    fromlist=firedrake.utility_meshes.__all__)
for name in firedrake.utility_meshes.__all__:
    setattr(thismod, name, getattr(meshes, name))
    __all__.append(name)
