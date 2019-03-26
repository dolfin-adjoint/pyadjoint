import backend
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import stop_annotating


def refine(*args, **kwargs):
    """ Refine is overloaded to ensure that the returned mesh is overloaded.
    """
    with stop_annotating():
        output = backend.refine(*args, **kwargs)
    overloaded = create_overloaded_object(output)
    return overloaded
