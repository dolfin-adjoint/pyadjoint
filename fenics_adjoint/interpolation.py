import backend
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import stop_annotating


def interpolate(*args, **kwargs):
    """Interpolation is overloaded to ensure that the returned Function object is overloaded.
    We are not able to annotate the interpolation call at the moment.

    """
    with stop_annotating():
        output = backend.interpolate(*args, **kwargs)
    return create_overloaded_object(output)
