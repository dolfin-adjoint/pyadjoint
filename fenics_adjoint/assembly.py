import backend
import ufl
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from pyadjoint.block import Block
from .types import create_overloaded_object, compat


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
        mesh = self.form.ufl_domain().ufl_cargo()
        self.add_dependency(mesh.block_variable)
        for c in self.form.coefficients():
            self.add_dependency(c.block_variable)

    def __str__(self):
        return str(self.form)

    @no_annotations
    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return

        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep

        form = ufl.replace(self.form, replaced_coeffs)

        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = replaced_coeffs.get(c, c)

            if isinstance(c, backend.Expression) or \
               isinstance(c, backend.CompiledExpression):
                # Create a FunctionSpace from self.form and Expression.
                # And then make a TestFunction from this space.
                mesh = self.form.ufl_domain().ufl_cargo()
                V = c._ad_function_space(mesh)
                dc = backend.TestFunction(V)

                dform = backend.derivative(form, c_rep, dc)
                output = compat.assemble_adjoint_value(dform)
                block_variable.add_adj_output([[adj_input * output, V]])
                continue

            elif isinstance(c, backend.Mesh):
                X = backend.SpatialCoordinate(c_rep)
                dform = backend.derivative(form, X)
                output = compat.assemble_adjoint_value(dform)
                block_variable.add_adj_output(adj_input * output)
                continue

            if isinstance(c, backend.Function):
                dc = backend.TestFunction(c.function_space())
            elif isinstance(c, backend.Constant):
                mesh = compat.extract_mesh_from_form(self.form)
                dc = backend.TestFunction(c._ad_function_space(mesh))

            dform = backend.derivative(form, c_rep, dc)
            output = compat.assemble_adjoint_value(dform)
            block_variable.add_adj_output(adj_input * output)

    @no_annotations
    def evaluate_tlm(self):
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep

        form = ufl.replace(self.form, replaced_coeffs)

        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = replaced_coeffs.get(c, c)
            tlm_value = block_variable.tlm_value

            if tlm_value is None:
                continue

            if isinstance(c, backend.Function):
                dform = backend.derivative(form, c_rep, tlm_value)
                output = compat.assemble_adjoint_value(dform)
                self.get_outputs()[0].add_tlm_output(output)

            elif isinstance(c, backend.Constant):
                dform = backend.derivative(form, c_rep, tlm_value)
                output = compat.assemble_adjoint_value(dform)
                self.get_outputs()[0].add_tlm_output(output)

            elif isinstance(c, backend.Expression):
                dform = backend.derivative(form, c_rep, tlm_value)
                output = compat.assemble_adjoint_value(dform)
                self.get_outputs()[0].add_tlm_output(output)

            elif isinstance(c, backend.Mesh):
                X = backend.SpatialCoordinate(c_rep)
                dform = backend.derivative(form, X, tlm_value)
                output = compat.assemble_adjoint_value(dform)
                self.get_outputs()[0].add_tlm_output(output)

    @no_annotations
    def evaluate_hessian(self):
        hessian_input = self.get_outputs()[0].hessian_value
        adj_input = self.get_outputs()[0].adj_value

        if hessian_input is None:
            return

        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep

        form = ufl.replace(self.form, replaced_coeffs)

        for bo1 in self.get_dependencies():
            c1 = bo1.output
            c1_rep = replaced_coeffs.get(c1, c1)

            if isinstance(c1, backend.Function):
                dc = backend.TestFunction(c1.function_space())
            elif isinstance(c1, backend.Expression):
                mesh = form.ufl_domain().ufl_cargo()
                W = c1._ad_function_space(mesh)
                dc = backend.TestFunction(W)
            elif isinstance(c1, backend.Constant):
                mesh = compat.extract_mesh_from_form(form)
                dc = backend.TestFunction(c1._ad_function_space(mesh))
            elif isinstance(c1, backend.Mesh):
                pass
            else:
                continue

            # Special treatment if CoordinateDerivative
            if isinstance(c1, backend.Mesh):
                X = backend.SpatialCoordinate(c1)
                dform = backend.derivative(form, X)
            else:
                dform = backend.derivative(form, c1_rep, dc)

            for bo2 in self.get_dependencies():
                c2 = bo2.output
                c2_rep = replaced_coeffs.get(c2, c2)
                tlm_input = bo2.tlm_value

                if tlm_input is None:
                    continue

                if isinstance(c2, backend.Mesh):
                    X = backend.SpatialCoordinate(c2)
                    ddform = backend.derivative(dform, X, tlm_input)
                else:
                    ddform = backend.derivative(dform, c2_rep, tlm_input)

                output = compat.assemble_adjoint_value(ddform)

                if isinstance(c1, backend.Function):
                    bo1.add_hessian_output(adj_input*output)
                elif isinstance(c1, backend.Expression):
                    bo1.add_hessian_output([(adj_input * output, W)])
                elif isinstance(c1, backend.Constant):
                    bo1.add_hessian_output(adj_input*output)
                elif isinstance(c1, backend.Mesh):
                    bo1.add_hessian_output(adj_input*output)
                else:
                    continue

            output = compat.assemble_adjoint_value(dform)
            if isinstance(c1, backend.Expression):
                bo1.add_hessian_output([(hessian_input*output, W)])
            else:
                bo1.add_hessian_output(hessian_input*output)

    @no_annotations
    def recompute(self):
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep

        form = ufl.replace(self.form, replaced_coeffs)

        output = backend.assemble(form)
        output = create_overloaded_object(output)

        self.get_outputs()[0].checkpoint = output._ad_create_checkpoint()
        # TODO: Assemble might output a float (AdjFloat), but this is NOT treated as
        #       a Coefficient by UFL and so the correct dependencies are not setup for Solve/Assemble blocks if
        #       the AdjFloat is used directly in a form. Using it in a Constant before a form would also not help,
        #       because the Constant is always seen as a leaf node, and thus can't depend on this AdjFloat.
        #       Right now the only workaround is using it in an Expression.
