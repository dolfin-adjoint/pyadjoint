import backend
import ufl
from pyadjoint import Block, create_overloaded_object

from dolfin_adjoint_common import compat
compat = compat.compat(backend)


class AssembleVectorBlock(Block):
    def __init__(self, form):
        super(AssembleVectorBlock, self).__init__()
        self.form = form
        mesh = self.form.ufl_domain().ufl_cargo()
        self.add_dependency(mesh)
        for c in self.form.coefficients():
            self.add_dependency(c, no_duplicates=True)

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
            dc = backend.TrialFunction(c.function_space())
        elif isinstance(c, backend.Constant):
            mesh = compat.extract_mesh_from_form(self.form)
            dc = backend.TrialFunction(c._ad_function_space(mesh))
        elif isinstance(c, compat.MeshType):
            c_rep = backend.SpatialCoordinate(c_rep)
            dc = backend.TrialFunction(c._ad_function_space())

        dform = backend.derivative(form, c_rep, dc)

        test_space = self.form.arguments()[0].function_space()
        adj_input_func = backend.Function(test_space)
        adj_input_func.vector()[:] = adj_input
        dform = backend.action(backend.adjoint(dform), adj_input_func)
        return compat.assemble_adjoint_value(dform)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form = prepared
        output = backend.assemble(form)
        output = create_overloaded_object(output)
        return output
