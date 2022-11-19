import ufl
from ufl.formatting.ufl2unicode import ufl2unicode
from pyadjoint import Block, create_overloaded_object


class AssembleBlock(Block):
    def __init__(self, form, ad_block_tag=None):
        super(AssembleBlock, self).__init__(ad_block_tag=ad_block_tag)
        self.form = form
        if self.backend.__name__ != "firedrake":
            mesh = self.form.ufl_domain().ufl_cargo()
        else:
            mesh = self.form.ufl_domain() if hasattr(self.form, 'ufl_domain') else None

        if mesh:
            self.add_dependency(mesh)

        for c in self.form.coefficients():
            self.add_dependency(c, no_duplicates=True)

    def __str__(self):
        return f"assemble({ufl2unicode(self.form)})"

    def compute_action_adjoint(self, adj_input, arity_form, form=None, c_rep=None, space=None, dform=None):
        if arity_form == 0:
            if dform is None:
                dc = self.backend.TestFunction(space)
                dform = self.backend.derivative(form, c_rep, dc)
            dform_vector = self.compat.assemble_adjoint_value(dform)
            # Return a Vector scaled by the scalar `adj_input`
            return dform_vector * adj_input, dform
        elif arity_form == 1:
            if dform is None:
                dc = self.backend.TrialFunction(space)
                dform = self.backend.derivative(form, c_rep, dc)
            # Get the Cofunction
            adj_input = adj_input.function
            # Symbolically compute: (dform/dc_rep)^* * adj_input
            adj_output = ufl.action(ufl.adjoint(dform), adj_input)
            return self.compat.assemble_adjoint_value(adj_output), dform
        else:
            raise ValueError('Forms with arity > 1 are not handled yet!')

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

        from ufl.algorithms.analysis import extract_arguments
        arity_form = len(extract_arguments(form))

        if isinstance(c, self.compat.ExpressionType):
            # Create a FunctionSpace from self.form and Expression.
            # And then make a TestFunction from this space.
            mesh = self.form.ufl_domain().ufl_cargo()
            V = c._ad_function_space(mesh)
            dc = self.backend.TestFunction(V)

            dform = self.backend.derivative(form, c_rep, dc)
            output = self.compat.assemble_adjoint_value(dform)
            return [[adj_input * output, V]]

        if isinstance(c, self.backend.Function):
            space = c.function_space()
        elif isinstance(c, self.backend.Constant):
            mesh = self.compat.extract_mesh_from_form(self.form)
            space = c._ad_function_space(mesh)
        elif isinstance(c, self.compat.MeshType):
            c_rep = self.backend.SpatialCoordinate(c_rep)
            space = c._ad_function_space()

        return self.compute_action_adjoint(adj_input, arity_form, form, c_rep, space)[0]

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, tlm_inputs, self.get_dependencies())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        form = prepared
        dform = 0.
        from ufl.algorithms.analysis import extract_arguments
        arity_form = len(extract_arguments(form))

        for bv in self.get_dependencies():
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value

            if tlm_value is None:
                continue
            if isinstance(c_rep, self.compat.MeshType):
                X = self.backend.SpatialCoordinate(c_rep)
                dform += self.backend.derivative(form, X, tlm_value)
            else:
                dform += self.backend.derivative(form, c_rep, tlm_value)
        if not isinstance(dform, float):
            dform = ufl.algorithms.expand_derivatives(dform)
            dform = self.compat.assemble_adjoint_value(dform)
            if arity_form == 1 and dform != 0:
                # Then dform is a Vector (and not a ZeroBaseForm since dform != 0)
                dform = dform.function
        return dform

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, adj_inputs, relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        form = prepared
        hessian_input = hessian_inputs[0]
        adj_input = adj_inputs[0]

        from ufl.algorithms.analysis import extract_arguments
        arity_form = len(extract_arguments(form))

        c1 = block_variable.output
        c1_rep = block_variable.saved_output

        if isinstance(c1, self.backend.Function):
            space = c1.function_space()
        elif isinstance(c1, self.compat.ExpressionType):
            mesh = form.ufl_domain().ufl_cargo()
            space = c1._ad_function_space(mesh)
        elif isinstance(c1, self.backend.Constant):
            mesh = self.compat.extract_mesh_from_form(form)
            space = c1._ad_function_space(mesh)
        elif isinstance(c1, self.compat.MeshType):
            c1_rep = self.backend.SpatialCoordinate(c1)
            space = c1._ad_function_space()
        else:
            return None

        hessian_outputs, dform = self.compute_action_adjoint(hessian_input, arity_form, form, c1_rep, space)

        ddform = 0.
        for other_idx, bv in relevant_dependencies:
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if isinstance(c2_rep, self.compat.MeshType):
                X = self.backend.SpatialCoordinate(c2_rep)
                ddform += self.backend.derivative(dform, X, tlm_input)
            else:
                ddform += self.backend.derivative(dform, c2_rep, tlm_input)

        if not isinstance(ddform, float):
            ddform = ufl.algorithms.expand_derivatives(ddform)
            if not (isinstance(ddform, ufl.ZeroBaseForm) or (isinstance(ddform, ufl.Form) and ddform.empty())):
                hessian_outputs += self.compute_action_adjoint(adj_input, arity_form, dform=ddform)[0]

        if isinstance(c1, self.compat.ExpressionType):
            return [(hessian_outputs, space)]
        else:
            return hessian_outputs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form = prepared
        output = self.backend.assemble(form)
        output = create_overloaded_object(output)
        return output
