import backend
from .types import create_overloaded_object

def refine(*args, **kwargs):
    output = backend.refine(*args, **kwargs)
    return create_overloaded_object(output)
