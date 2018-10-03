import backend
from pyadjoint.overloaded_type import create_overloaded_object


def interpolate(*args, **kwargs):
    output = backend.interpolate(*args, **kwargs)
    return create_overloaded_object(output)