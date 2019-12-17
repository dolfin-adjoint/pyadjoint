from pyadjoint import Block

class ExpressionBlock(Block):
    def __init__(self, expression):
        super(ExpressionBlock, self).__init__()
        self.expression = expression
        self.dependency_keys = {}

        for key in expression._ad_attributes_dict:
            parameter = expression._ad_attributes_dict[key]
            if isinstance(parameter, OverloadedType):
                self.add_dependency(parameter)
                self.dependency_keys[parameter] = key

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_inputs = adj_inputs[0]
        c = block_variable.output
        if c not in self.expression.user_defined_derivatives:
            return None

        for key in self.expression._ad_attributes_dict:
            if key not in self.expression.ad_ignored_attributes:
                setattr(self.expression.user_defined_derivatives[c], key,
                        self.expression._ad_attributes_dict[key])

        adj_output = None
        for adj_pair in adj_inputs:
            adj_input = adj_pair[0]
            V = adj_pair[1]
            if adj_output is None:
                adj_output = 0.0

            interp = self.compat.interpolate(self.expression.user_defined_derivatives[c], V)
            if isinstance(c, (self.compat.Constant, AdjFloat)):
                adj_output += adj_input.inner(interp.vector())
            else:
                vec = adj_input * interp.vector()
                adj_func = self.compat.Function(V, vec)

                num_sub_spaces = V.num_sub_spaces()
                if num_sub_spaces > 1:
                    for i in range(num_sub_spaces):
                        adj_output += self.compat.interpolate(adj_func.sub(i), c.function_space()).vector()
                else:
                    adj_output += self.compat.interpolate(adj_func, c.function_space()).vector()
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        # Restore _ad_attributes_dict.
        block_variable.saved_output

        tlm_output = 0.
        tlm_used = False
        for block_variable in self.get_dependencies():
            tlm_input = block_variable.tlm_value
            if tlm_input is None:
                continue

            c = block_variable.output
            if c not in self.expression.user_defined_derivatives:
                continue

            for key in self.expression._ad_attributes_dict:
                if key not in self.expression.ad_ignored_attributes:
                    setattr(self.expression.user_defined_derivatives[c], key, self.expression._ad_attributes_dict[key])

            tlm_used = True
            tlm_output += tlm_input * self.expression.user_defined_derivatives[c]
        if not tlm_used:
            return None
        return tlm_output

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        hessian_inputs = hessian_inputs[0]
        adj_inputs = adj_inputs[0]
        c1 = block_variable.output

        if c1 not in self.expression.user_defined_derivatives:
            return None

        first_deriv = self.expression.user_defined_derivatives[c1]
        for key in self.expression._ad_attributes_dict:
            if key not in self.expression.ad_ignored_attributes:
                setattr(first_deriv, key, self.expression._ad_attributes_dict[key])

        hessian_output = None
        for _, bo2 in relevant_dependencies:
            c2 = bo2.output
            tlm_input = bo2.tlm_value

            if tlm_input is None:
                continue

            if c2 not in first_deriv.user_defined_derivatives:
                continue

            second_deriv = first_deriv.user_defined_derivatives[c2]
            for key in self.expression._ad_attributes_dict:
                if key not in self.expression.ad_ignored_attributes:
                    setattr(second_deriv, key, self.expression._ad_attributes_dict[key])

            for adj_pair in adj_inputs:
                adj_input = adj_pair[0]
                V = adj_pair[1]

                if hessian_output is None:
                    hessian_output = 0.0

                # TODO: Seems we can only project and not interpolate ufl.algebra.Product in dolfin.
                #       Consider the difference and which actually makes sense here.
                interp = self.compat.project(tlm_input * second_deriv, V)
                if isinstance(c1, (self.compat.Constant, AdjFloat)):
                    hessian_output += adj_input.inner(interp.vector())
                else:
                    vec = adj_input * interp.vector()
                    hessian_func = self.compat.Function(V, vec)

                    num_sub_spaces = V.num_sub_spaces()
                    if num_sub_spaces > 1:
                        for i in range(num_sub_spaces):
                            hessian_output += self.compat.interpolate(hessian_func.sub(i), c1.function_space()).vector()
                    else:
                        hessian_output += self.compat.interpolate(hessian_func, c1.function_space()).vector()

        for hessian_pair in hessian_inputs:
            if hessian_output is None:
                hessian_output = 0.0
            hessian_input = hessian_pair[0]
            V = hessian_pair[1]

            interp = self.compat.interpolate(first_deriv, V)
            if isinstance(c1, (self.compat.Constant, AdjFloat)):
                hessian_output += hessian_input.inner(interp.vector())
            else:
                vec = hessian_input * interp.vector()
                hessian_func = self.compat.Function(V, vec)

                num_sub_spaces = V.num_sub_spaces()
                if num_sub_spaces > 1:
                    for i in range(num_sub_spaces):
                        hessian_output += self.compat.interpolate(hessian_func.sub(i), c1.function_space()).vector()
                else:
                    hessian_output += self.compat.interpolate(hessian_func, c1.function_space()).vector()
        return hessian_output

    def recompute_component(self, inputs, block_variable, idx, prepared):
        checkpoint = self.get_outputs()[0].checkpoint

        if checkpoint:
            for bv in self.get_dependencies():
                key = self.dependency_keys[bv.output]
                checkpoint[key] = bv.saved_output

    def __str__(self):
        return "Expression block"
