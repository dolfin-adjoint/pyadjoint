import backend
import ufl
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape
from pyadjoint.block import Block
from pyadjoint.overloaded_type import create_overloaded_object
from .types import compat
from .formmanipulations import UFLForm, UFLFormCoefficient


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
    A.assemble_system = True

    return A, b


class AssembleBlock(Block):
    def __init__(self, form):
        super(AssembleBlock, self).__init__()
        self.form = form
        if backend.__name__ != "firedrake":
            mesh = self.form.ufl_domain().ufl_cargo()
        else:
            mesh = self.form.ufl_domain()
        self.add_dependency(mesh)
        for c in self.form.coefficients():
            self.add_dependency(c, no_duplicates=True)
        self._ufl_form = None

    def __str__(self):
        return str(self.form)

    @property
    def ufl_form(self):
        if self._ufl_form is None:
            self._ufl_form = self.create_ufl_form()
        return self._ufl_form

    def create_ufl_form(self):
        replaced_coeffs = self.replace_map()
        return UFLForm.create(self.form, replaced_coeffs)

    def replace_map(self):
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep
        return replaced_coeffs

    def replaced_form(self):
        replaced_coeffs = self.replace_map()
        self.ufl_form.replace(replaced_coeffs)
        return self.ufl_form

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        return self.replaced_form()

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

            dform = form.derivative(c, dc)
            output = compat.assemble_adjoint_value(dform)
            return [[adj_input * output, V]]
        elif isinstance(c, compat.MeshType):
            X = backend.SpatialCoordinate(c_rep)
            dform = form.derivative(X)
            output = compat.assemble_adjoint_value(dform)
            return adj_input * output

        if isinstance(c, backend.Function):
            dc = backend.TestFunction(c.function_space())
        elif isinstance(c, backend.Constant):
            mesh = compat.extract_mesh_from_form(self.form)
            dc = backend.TestFunction(c._ad_function_space(mesh))

        dform = form.derivative(c, dc)
        output = compat.assemble_adjoint_value(dform)
        return adj_input * output

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self.replaced_form()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        form = prepared
        dform = 0.
        dform_shape = 0.
        for bv in self.get_dependencies():
            c = bv.output
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value

            if tlm_value is None:
                continue
            if isinstance(c_rep, compat.MeshType):
                X = backend.SpatialCoordinate(c_rep)
                dform_shape += compat.assemble_adjoint_value(
                    form.derivative(X, tlm_value))
            else:
                dform += form.derivative(c, tlm_value)
        if not isinstance(dform, float):
            dform = compat.assemble_adjoint_value(dform)
        return dform + dform_shape

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self.replaced_form()

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        form = prepared
        hessian_input = hessian_inputs[0]
        adj_input = adj_inputs[0]

        c1 = block_variable.output
        c1_rep = block_variable.saved_output

        if isinstance(c1, backend.Function):
            dc = backend.TestFunction(c1.function_space())
        elif isinstance(c1, compat.ExpressionType):
            mesh = form.ufl_domain().ufl_cargo()
            W = c1._ad_function_space(mesh)
            dc = backend.TestFunction(W)
        elif isinstance(c1, backend.Constant):
            mesh = compat.extract_mesh_from_form(form)
            dc = backend.TestFunction(c1._ad_function_space(mesh))
        elif isinstance(c1, compat.MeshType):
            pass
        else:
            return None

        if isinstance(c1, compat.MeshType):
            X = backend.SpatialCoordinate(c1)
            dform = form.derivative(X)
        else:
            dform = form.derivative(c1, dc)
        hessian_outputs = hessian_input * compat.assemble_adjoint_value(dform)

        for other_idx, bv in relevant_dependencies:
            c2 = bv.output
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if isinstance(c2_rep, compat.MeshType):
                X = backend.SpatialCoordinate(c2_rep)
                ddform = dform.derivative(X, tlm_input)
            else:
                ddform = dform.derivative(c2, tlm_input)

            if not ddform.empty():
                hessian_outputs += adj_input * compat.assemble_adjoint_value(ddform)

        if isinstance(c1, compat.ExpressionType):
            return [(hessian_outputs, W)]
        else:
            return hessian_outputs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.replaced_form()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form = prepared
        output = backend.assemble(form)
        output = create_overloaded_object(output)
        return output
