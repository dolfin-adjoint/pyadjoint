import fenics_adjoint.types.function as function
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from fenics_adjoint.projection import ProjectBlock
from fenics_adjoint.types import create_overloaded_object


class Function(function.Function):

    def project(self, b, *args, **kwargs):
        annotate = annotate_tape(kwargs)
        with stop_annotating():
            output = super(Function, self).project(b, *args, **kwargs)
        output = create_overloaded_object(output)

        if annotate:
            bcs = kwargs.pop("bcs", [])
            block = ProjectBlock(b, self.function_space(), output, bcs)

            tape = get_working_tape()
            tape.add_block(block)

            block.add_output(output.get_block_output())

        return output
