import backend
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating
from pyadjoint.block import Block
from pyadjoint.overloaded_type import create_overloaded_object
from .solving import SolveBlock


def project(*args, **kwargs):
    '''The project call performs an equation solve, and so it too must be annotated so that the
    adjoint and tangent linear models may be constructed automatically by pyadjoint.

    To disable the annotation of this function, just pass :py:data:`annotate=False`. This is useful in
    cases where the solve is known to be irrelevant or diagnostic for the purposes of the adjoint
    computation (such as projecting fields to other function spaces for the purposes of
    visualisation).'''

    annotate = annotate_tape(kwargs)
    with stop_annotating():
        output = backend.project(*args, **kwargs)
    output = create_overloaded_object(output)

    if annotate:
        bcs = kwargs.pop("bcs", [])
        block = ProjectBlock(args[0], args[1], output, bcs)

        tape = get_working_tape()
        tape.add_block(block)

        block.add_output(output.block_variable)

    return output


class ProjectBlock(SolveBlock):
    def __init__(self, v, V, output, bcs=[]):
        w = backend.TestFunction(V)
        Pv = backend.TrialFunction(V)
        a = backend.inner(w, Pv)*backend.dx
        L = backend.inner(w, v)*backend.dx

        super(ProjectBlock, self).__init__(a == L, output, bcs)
