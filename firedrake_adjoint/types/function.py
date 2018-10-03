import backend
import fenics_adjoint.types.function as function
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from fenics_adjoint.projection import ProjectBlock
from pyadjoint.overloaded_type import create_overloaded_object, register_overloaded_type


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

            block.add_output(output.create_block_variable())

        return output

    def split(self):
        """Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        # TODO: This should produce some kind of annotation.
        if self._split is None:
            self._split = tuple(Function(fs, dat, name="%s[%d]" % (self.name(), i))
                                for i, (fs, dat) in
                                enumerate(zip(self.function_space(), self.dat)))
        return self._split

register_overloaded_type(Function, backend.Function)