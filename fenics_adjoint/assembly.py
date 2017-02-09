import backend
from .tape import Tape, Block, get_working_tape

def assemble(*args, **kwargs):
    tape = get_working_tape()

    annotate_tape = kwargs.pop("annotate_tape", True)
    output = backend.assemble(*args, **kwargs)

    if annotate_tape:
        form = args[0]

        block = AssembleBlock(form)
        tape.add_block(block)

        output = block.create_reference_object(output)

    return output



class AssembleBlock(Block):
    def __init__(self, form):
        super(AssembleBlock, self).__init__()
        self.form = form
        for c in self.form.coefficients():
            self.add_dependency(c)

    def evaluate_adj(self):
        adj_input = self.fwd_outputs[0].get_adj_output()

        for c in self.get_dependencies():
            if isinstance(c, backend.Function):
                dc = backend.TestFunction(c.function_space())
            elif isinstance(c, backend.Constant):
                dc = backend.Constant(1)

            dform = backend.derivative(self.form, c, dc)
            output = backend.assemble(dform)
            c.add_adj_output(adj_input * output)