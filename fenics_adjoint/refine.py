import backend
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import stop_annotating


__backend_refine = backend.refine


def refine(*args, **kwargs):
    """ Refine is overloaded to ensure that the returned mesh is overloaded.
    """
    with stop_annotating():
        output = __backend_refine(*args, **kwargs)
    overloaded = create_overloaded_object(output)
    return overloaded
