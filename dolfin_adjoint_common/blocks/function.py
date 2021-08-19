import ufl
from ufl.corealg.traversal import traverse_unique_terminals
from pyadjoint import Block, OverloadedType, AdjFloat


class FunctionAssignBlock(Block):
    def __init__(self, func, other, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.other = None
        self.expr = None
        if isinstance(other, float) or isinstance(other, int):
            other = AdjFloat(other)
        if isinstance(other, OverloadedType):
            self.add_dependency(other, no_duplicates=True)
        elif not(isinstance(other, float) or isinstance(other, int)):
            # Assume that this is a point-wise evaluated UFL expression (firedrake only)
            for op in traverse_unique_terminals(other):
                if isinstance(op, OverloadedType):
                    self.add_dependency(op, no_duplicates=True)
            self.expr = other

    def _replace_with_saved_output(self):
        if self.expr is None:
            return None

        replace_map = {}
        for dep in self.get_dependencies():
            replace_map[dep.output] = dep.saved_output
        return ufl.replace(self.expr, replace_map)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        V = self.get_outputs()[0].output.function_space()
        adj_input_func = self.compat.function_from_vector(V, adj_inputs[0])

        if self.expr is None:
            return adj_input_func

        expr = self._replace_with_saved_output()
        return expr, adj_input_func

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if self.expr is None:
            if isinstance(block_variable.output, AdjFloat):
                return adj_inputs[0].sum()
            elif isinstance(block_variable.output, self.backend.Constant):
                R = block_variable.output._ad_function_space(prepared.function_space().mesh())
                return self._adj_assign_constant(prepared, R)
            else:
                adj_output = self.backend.Function(
                    block_variable.output.function_space())
                adj_output.assign(prepared)
                return adj_output.vector()
        else:
            # Linear combination
            expr, adj_input_func = prepared
            adj_output = self.backend.Function(adj_input_func.function_space())
            if block_variable.saved_output.ufl_shape == adj_input_func.ufl_shape:
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output, adj_input_func)
                )
                adj_output.assign(diff_expr)
            else:
                # Assume current variable is scalar constant.
                # This assumption might be wrong for firedrake.
                assert isinstance(block_variable.output, self.backend.Constant)

                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output, self.backend.Constant(1.))
                )
                adj_output.assign(diff_expr)
                return adj_output.vector().inner(adj_input_func.vector())

            if isinstance(block_variable.output, self.backend.Constant):
                R = block_variable.output._ad_function_space(adj_output.function_space().mesh())
                return self._adj_assign_constant(adj_output, R)
            else:
                return adj_output.vector()

    def _adj_assign_constant(self, adj_output, constant_fs):
        r = self.backend.Function(constant_fs)
        shape = r.ufl_shape
        if shape == () or shape[0] == 1:
            # Scalar Constant
            r.vector()[:] = adj_output.vector().sum()
        else:
            # We assume the shape of the constant == shape of the output function if not scalar.
            # This assumption is due to FEniCS not supporting products with non-scalar constants in assign.
            values = []
            for i in range(shape[0]):
                values.append(adj_output.sub(i, deepcopy=True).vector().sum())
            r.assign(self.backend.Constant(values))
        return r.vector()

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        if self.expr is None:
            return None

        return self._replace_with_saved_output()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        if self.expr is None:
            return tlm_inputs[0]

        expr = prepared
        dudm = self.backend.Function(block_variable.output.function_space())
        dudmi = self.backend.Function(block_variable.output.function_space())
        for dep in self.get_dependencies():
            if dep.tlm_value:
                dudmi.assign(ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, dep.saved_output,
                                   dep.tlm_value)))
                dudm.vector().axpy(1.0, dudmi.vector())

        return dudm

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
        if self.expr is None:
            return None
        return self._replace_with_saved_output()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        if self.expr is None:
            prepared = inputs[0]
        output = self.backend.Function(block_variable.output.function_space())
        self.backend.Function.assign(output, prepared)
        return output
