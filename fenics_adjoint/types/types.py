import backend
from .function import Function
from .constant import Constant
from .mesh import Mesh
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.overloaded_type import OverloadedType


def create_overloaded_object(obj,**kwargs):
    """Creates an :class:`OverloadedType` instance corresponding `obj`.

    Args:
        obj: The object to create an overloaded object from.

    Returns:
        :class:`OverloadedType`: An object which has the same attributes as `obj`, but also the extra attributes/methods needed for use in the tape.

    Raises:
        NotImplemntedError: If the corresponding :class:`OverloadedType` has not been implemented.

    """
    if isinstance(obj, OverloadedType):
        return obj
    if isinstance(obj, float):
        return AdjFloat(obj)
    elif isinstance(obj, backend.Function):
        # This will invoke the backend constructor in a way that is said to be only intended for internal library use. 

        return Function(obj.function_space(), obj.vector(),**kwargs)
    elif isinstance(obj, backend.Constant):
        return Constant(obj)
    elif isinstance(obj, backend.Mesh):
        return Mesh(obj)
    else:
        raise NotImplementedError
