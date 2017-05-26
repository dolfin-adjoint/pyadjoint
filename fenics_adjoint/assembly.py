import backend
import ufl
from pyadjoint.tape import get_working_tape
from pyadjoint.block import Block
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

    def __str__(self):
        return str(self.form)

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()

        replaced_coeffs = {}
        for block_output in self.get_dependencies():
            coeff = block_output.get_output()
            replaced_coeffs[coeff] = block_output.get_saved_output()

        form = backend.replace(self.form, replaced_coeffs)

        for block_output in self.get_dependencies():
            c = block_output.get_output()
            c_rep = replaced_coeffs.get(c, c)

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

    def evaluate_tlm(self):
        replaced_coeffs = {}
        for block_output in self.get_dependencies():
            coeff = block_output.get_output()
            replaced_coeffs[coeff] = block_output.get_saved_output()

        form = backend.replace(self.form, replaced_coeffs)

        for block_output in self.get_dependencies():
            c = block_output.get_output()
            c_rep = replaced_coeffs.get(c, c)
            tlm_value = block_output.tlm_value

            if tlm_value is None:
                continue

            if isinstance(c, backend.Function):
                dform = backend.derivative(form, c_rep, tlm_value)
                output = backend.assemble(dform)
                self.get_outputs()[0].add_tlm_output(output)

            elif isinstance(c, backend.Constant):
                dform = backend.derivative(form, c_rep, tlm_value)
                output = backend.assemble(dform)
                self.get_outputs()[0].add_tlm_output(output)

            elif isinstance(c, backend.Expression):
                dform = backend.derivative(form, c_rep, tlm_value)
                output = backend.assemble(dform)
                self.get_outputs()[0].add_tlm_output(output)

    def evaluate_hessian(self):
        hessian_input = self.get_outputs()[0].hessian_value
        adj_input = self.get_outputs()[0].adj_value

        replaced_coeffs = {}
        for block_output in self.get_dependencies():
            coeff = block_output.get_output()
            replaced_coeffs[coeff] = block_output.get_saved_output()

        form = backend.replace(self.form, replaced_coeffs)

        for bo1 in self.get_dependencies():
            c1 = bo1.get_output()
            c1_rep = replaced_coeffs.get(c1, c1)

            if isinstance(c1, backend.Function):
                dc = backend.TestFunction(c1.function_space())
            elif isinstance(c1, backend.Constant):
                dc = backend.Constant(1)
            else:
                continue

            dform = backend.derivative(form, c1_rep, dc)

            for bo2 in self.get_dependencies():
                c2 = bo2.get_output()
                c2_rep = replaced_coeffs.get(c2, c2)
                tlm_input = bo2.tlm_value

                if tlm_input is None:
                    continue

                if isinstance(c1, backend.Function):
                    ddform = backend.derivative(dform, c2_rep, tlm_input)
                    output = backend.assemble(ddform)
                    bo1.add_hessian_output(adj_input*output)
                elif isinstance(c1, backend.Constant):
                    ddform = backend.derivative(dform, c2_rep, tlm_input)
                    output = backend.assemble(ddform)
                    bo1.add_hessian_output(adj_input*output)
                else:
                    continue

            output = backend.assemble(dform)
            bo1.add_hessian_output(hessian_input*output)

    def recompute(self):
        replaced_coeffs = {}
        for block_output in self.get_dependencies():
            coeff = block_output.get_output()
            replaced_coeffs[coeff] = block_output.get_saved_output()

        form = backend.replace(self.form, replaced_coeffs)

        output = backend.assemble(form)
        output = create_overloaded_object(output)

        self.get_outputs()[0].checkpoint = output._ad_create_checkpoint()
        # TODO: Assemble might output a float (AdjFloat), but this is NOT treated as
        #       a Coefficient by UFL and so the correct dependencies are not setup for Solve/Assemble blocks if
        #       the AdjFloat is used directly in a form. Using it in a Constant before a form would also not help,
        #       because the Constant is always seen as a leaf node, and thus can't depend on this AdjFloat.
        #       Right now the only workaround is using it in an Expression.
