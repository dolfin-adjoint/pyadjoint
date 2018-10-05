import backend
import ufl
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from pyadjoint.block import Block
from pyadjoint.overloaded_type import create_overloaded_object
from .types import compat


def assemble(*args, **kwargs):
    """When a form is assembled, the information about its nonlinear dependencies is lost,
    and it is no longer easy to manipulate. Therefore, fenics_adjoint overloads the :py:func:`dolfin.assemble`
    function to *attach the form to the assembled object*. This lets the automatic annotation work,
    even when the user calls the lower-level :py:data:`solve(A, x, b)`.
    """
    annotate = annotate_tape(kwargs)
    with stop_annotating():
        output = backend.assemble(*args, **kwargs)

    form = args[0]
    if isinstance(output, float):
        output = create_overloaded_object(output)

        if annotate:
            block = AssembleBlock(form)

            tape = get_working_tape()
            tape.add_block(block)

            block.add_output(output.block_variable)
    else:
        # Assembled a vector or matrix
        output.form = form

    return output

def assemble_system(*args, **kwargs):
    """When a form is assembled, the information about its nonlinear dependencies is lost,
    and it is no longer easy to manipulate. Therefore, fenics_adjoint overloads the :py:func:`dolfin.assemble_system`
    function to *attach the form to the assembled object*. This lets the automatic annotation work,
    even when the user calls the lower-level :py:data:`solve(A, x, b)`.
    """
    A_form = args[0]
    b_form = args[1]

    A, b = backend.assemble_system(*args, **kwargs)

    if "bcs" in kwargs:
        bcs = kwargs["bcs"]
    elif len(args) > 2:
        bcs = args[2]
    else:
        bcs = []

    A.form = A_form
    A.bcs = bcs
    b.form = b_form
    b.bcs = bcs

    return A, b

class AssembleBlock(Block):
    def __init__(self, form):
        super(AssembleBlock, self).__init__()
        self.form = form
        for c in self.form.coefficients():
            self.add_dependency(c.block_variable, no_duplicates=True)

    def __str__(self):
        return str(self.form)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep

        form = ufl.replace(self.form, replaced_coeffs)
        return form

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        form = prepared
        adj_input = adj_inputs[0]
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, compat.ExpressionType):
            # Create a FunctionSpace from self.form and Expression.
            # And then make a TestFunction from this space.
            mesh = self.form.ufl_domain().ufl_cargo()
            V = c._ad_function_space(mesh)
            dc = backend.TestFunction(V)

            dform = backend.derivative(form, c_rep, dc)
            output = compat.assemble_adjoint_value(dform)
            return [[adj_input * output, V]]

        if isinstance(c, backend.Function):
            dc = backend.TestFunction(c.function_space())
        elif isinstance(c, backend.Constant):
            mesh = compat.extract_mesh_from_form(self.form)
            dc = backend.TestFunction(c._ad_function_space(mesh))

        dform = backend.derivative(form, c_rep, dc)
        output = compat.assemble_adjoint_value(dform)
        return adj_input * output

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, tlm_inputs, self.get_dependencies())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        form = prepared
        dform = 0.
        for bv in self.get_dependencies():
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value

            if tlm_value is None:
                continue

            dform += backend.derivative(form, c_rep, tlm_value)
        return compat.assemble_adjoint_value(dform)

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, adj_inputs, relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        form = prepared
        hessian_input = hessian_inputs[0]
        adj_input = adj_inputs[0]

        c1 = block_variable.output
        c1_rep = block_variable.saved_output

        if isinstance(c1, backend.Function):
            dc = backend.TestFunction(c1.function_space())
        elif isinstance(c1, backend.Expression):
            mesh = form.ufl_domain().ufl_cargo()
            W = c1._ad_function_space(mesh)
            dc = backend.TestFunction(W)
        elif isinstance(c1, backend.Constant):
            mesh = compat.extract_mesh_from_form(form)
            dc = backend.TestFunction(c1._ad_function_space(mesh))
        else:
            return None

        dform = backend.derivative(form, c1_rep, dc)
        hessian_outputs = hessian_input*dform

        for other_idx, bv in relevant_dependencies:
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            ddform = backend.derivative(dform, c2_rep, tlm_input)
            hessian_outputs += adj_input*ddform

        hessian_output = compat.assemble_adjoint_value(hessian_outputs)
        if isinstance(c1, backend.Expression):
            return [(hessian_output, W)]
        else:
            return hessian_output

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form = prepared
        output = backend.assemble(form)
        output = create_overloaded_object(output)
        return output
