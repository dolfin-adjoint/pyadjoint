import backend
import ufl
from pyadjoint.tape import Block, get_working_tape
from .types import create_overloaded_object


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

        replaced_coeffs = {}
        for block_output in self.get_dependencies():
            coeff = block_output.get_output()
            replaced_coeffs[coeff] = block_output.get_saved_output()

        form = backend.replace(self.form, replaced_coeffs)

        for block_output in self.get_dependencies():
            c = block_output.get_output()
            if c in replaced_coeffs:
                c_rep = replaced_coeffs[c]
            else:
                c_rep = c

            if isinstance(c, backend.Expression):
                # Create a FunctionSpace from self.form and Expression.
                # And then make a TestFunction from this space.

                mesh = self.form.ufl_domain().ufl_cargo()
                c_element = c.ufl_element()

                # In newer versions of FEniCS there is a method named reconstruct, thus we may
                # in the future just call c_element.reconstruct(cell=mesh.ufl_cell()).
                element = ufl.FiniteElement(c_element.family(), mesh.ufl_cell(), c_element.degree())
                V = backend.FunctionSpace(mesh, element)
                dc = backend.TestFunction(V)

                dform = backend.derivative(form, c_rep, dc)
                output = backend.assemble(dform)
                block_output.add_adj_output([[adj_input * output, V]])

                continue

            if isinstance(c, backend.Function):
                dc = backend.TestFunction(c.function_space())
            elif isinstance(c, backend.Constant):
                dc = backend.Constant(1)

            dform = backend.derivative(form, c_rep, dc)
            output = backend.assemble(dform)
            block_output.add_adj_output(adj_input * output)