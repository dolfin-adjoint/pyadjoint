import backend
from .tape import Tape, Block, get_working_tape, create_overloaded_object

def assemble(*args, **kwargs):
    annotate_tape = kwargs.pop("annotate_tape", True)
    output = backend.assemble(*args, **kwargs)
    output = create_overloaded_object(output)

    if annotate_tape:
        form = args[0]
        block = AssembleBlock(form)

        tape = get_working_tape()
        tape.add_block(block)
        
        block.add_output(output.get_block_output())

    return output


class AssembleBlock(Block):
    def __init__(self, form):
        super(AssembleBlock, self).__init__()
        self.form = form
        for c in self.form.coefficients():
            self.add_dependency(c.get_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()

        for block_output in self.get_dependencies():
            c = block_output.get_output()
            if isinstance(c, backend.Function):
                dc = backend.TestFunction(c.function_space())
            elif isinstance(c, backend.Constant):
                dc = backend.Constant(1)

            dform = backend.derivative(self.form, c, dc)
            output = backend.assemble(dform)
            block_output.add_adj_output(adj_input * output)