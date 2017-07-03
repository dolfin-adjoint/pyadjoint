import backend
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating
from pyadjoint.block import Block
from .types import create_overloaded_object
from .solving import SolveBlock


def project(*args, **kwargs):
    annotate = annotate_tape(kwargs)
    with stop_annotating():
        output = backend.project(*args, **kwargs)
    output = create_overloaded_object(output)

    if annotate:
        bcs = kwargs.pop("bcs", [])
        block = ProjectBlock(args[0], args[1], output, bcs)

        tape = get_working_tape()
        tape.add_block(block)

        block.add_output(output.get_block_output())

    return output


class ProjectBlock(SolveBlock):
    def __init__(self, v, V, output, bcs=[]):
        w = backend.TestFunction(V)
        Pv = backend.TrialFunction(V)
        a = backend.inner(w, Pv)*backend.dx
        L = backend.inner(w, v)*backend.dx

        super(ProjectBlock, self).__init__(a == L, output, bcs)

    def recompute(self):
        SolveBlock.recompute(self)

