from pyadjoint import Block


class FunctionAssignBlock(Block):
    def __init__(self, func, other):
        super(AssignBlock, self).__init__()
        self.other = None
        self.lincom = False
        if isinstance(other, OverloadedType):
            self.add_dependency(other, no_duplicates=True)
        else:
            # Assume that this is a linear combination
            functions = _extract_functions_from_lincom(other)
            for f in functions:
                self.add_dependency(f, no_duplicates=True)
            self.expr = other
            self.lincom = True

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        V = self.get_outputs()[0].output.function_space()
        adj_input_func = compat.function_from_vector(V, adj_inputs[0])

        if not self.lincom:
            return adj_input_func
        # If what was assigned was not a lincom (only currently relevant in firedrake),
        # then we need to replace the coefficients in self.expr with new values.
        replace_map = {}
        for dep in self.get_dependencies():
            replace_map[dep.output] = dep.saved_output
        expr = ufl.replace(self.expr, replace_map)
        return expr, adj_input_func

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if not self.lincom:
            if isinstance(block_variable.output, (AdjFloat, self.compat.Constant)):
                return adj_inputs[0].sum()
            else:
                adj_output = self.compat.Function(
                    block_variable.output.function_space())
                adj_output.assign(prepared)
                return adj_output.vector()
        else:
            # Linear combination
            expr, adj_input_func = prepared
            adj_output = self.compat.Function(
                block_variable.output.function_space())
            diff_expr = ufl.algorithms.expand_derivatives(
                ufl.derivative(expr, block_variable.saved_output,
                               adj_input_func)
            )
            adj_output.assign(diff_expr)
            return adj_output.vector()

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        if not self.lincom:
            return None

        replace_map = {}
        for dep in self.get_dependencies():
            V = dep.output.function_space()
            tlm_input = dep.tlm_value or self.compat.Function(V)
            replace_map[dep.output] = tlm_input
        expr = ufl.replace(self.expr, replace_map)

        return expr

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        if not self.lincom:
            return tlm_inputs[0]

        expr = prepared
        V = block_variable.output.function_space()
        tlm_output = self.compat.Function(V)
        self.compat.Function.assign(tlm_output, expr)
        return tlm_output

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, hessian_inputs,
                                         relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        # Current implementation assumes lincom in hessian,
        # otherwise we need second-order derivatives here.
        return self.evaluate_adj_component(inputs, hessian_inputs,
                                           block_variable, idx, prepared)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        if not self.lincom:
            return None

        replace_map = {}
        for dep in self.get_dependencies():
            replace_map[dep.output] = dep.saved_output
        return ufl.replace(self.expr, replace_map)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        if not self.lincom:
            prepared = inputs[0]
        output = self.compat.Function(block_variable.output.function_space())
        self.compat.Function.assign(output, prepared)
        return output

