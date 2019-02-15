import backend
from pyadjoint.overloaded_type import create_overloaded_object


def refine(*args, **kwargs):
    """ Refine is overloaded to ensure that the returned mesh is overloaded.
    We are not able to annotate the interpolation call at the moment.
    """
    output = backend.refine(*args, **kwargs)
    return create_overloaded_object(output)
