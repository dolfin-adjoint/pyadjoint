import ufl
from ufl.corealg.traversal import traverse_unique_terminals
from pyadjoint import Block, OverloadedType, AdjFloat
from functools import reduce
import numpy


class FunctionAssignBlock(Block):
    def __init__(self, func, other):
        super().__init__()
        self.other = None
        self.expr = None
        if isinstance(other, OverloadedType):
            self.add_dependency(other, no_duplicates=True)
        elif not(isinstance(other, float) or isinstance(other, int)):
            # Assume that this is a point-wise evaluated UFL expression
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
        if isinstance(block_variable.output, self.backend.Constant):
            mesh = self.get_outputs()[0].output.ufl_domain().ufl_cargo()
            rspace = block_variable.output._ad_function_space(mesh)
            result = self.backend.Function(rspace)
            V = self.get_outputs()[0].output.function_space()
            dudm_func = self.backend.Function(V)
            shape = block_variable.output.ufl_shape
            if len(shape) > 0:
                if self.expr is not None:
                    # in fenics, none of the allowed UFL-expressions (linear combinations)
                    # allow for vector constants
                    # in firedrake, vector constants do not work with firedrake_adjoint at all
                    raise NotImplementedError("Vector constants in UFL expression assignment not implemented.")

                n = reduce(int.__mul__, shape)
                cvec = numpy.zeros(n)
                rvec = numpy.zeros(n)
                for i in range(n):
                    cvec[i] = 1
                    dudm_func.assign(self.backend.Constant(cvec.reshape(shape)))
                    rvec[i] = adj_inputs[0].inner(dudm_func.vector())
                    cvec[i] = 0

                result.assign(self.backend.Constant(rvec.reshape(shape)))
            else:
                if self.expr is None:
                    dudm_func.assign(self.backend.Constant(1))
                else:
                    expr, _ = prepared
                    dudm = ufl.algorithms.expand_derivatives(ufl.diff(expr, block_variable.saved_output))
                    dudm_func.assign(dudm)
                result.assign(self.backend.Constant(adj_inputs[0].inner(dudm_func.vector())))

            return result.vector()
        elif isinstance(block_variable.output, AdjFloat):
            assert self.expr is None, "AdjFloat not expected to occur in UFL expressions"
            adj_input_func = prepared
            dudm_func = self.backed.Function(adj_input_func.function_space())
            dudm_func.assign(1)
            return adj_inputs[0].inner(dudm_func.vector())
        else:
            if self.expr is None:
                adj_input_func = prepared
                dJdm = adj_input_func
            else:
                expr, adj_input_func = prepared
                if self.backend.__name__ != "firedrake":
                    # in fenics only simple linear combinations (i.e. scalar constant times vector function)
                    # are allowed, and thus the following should be safe
                    dJdm = ufl.algorithms.expand_derivatives(ufl.derivative(
                        expr, block_variable.saved_output, adj_input_func))
                else:
                    dudm = ufl.algorithms.expand_derivatives(ufl.diff(expr, block_variable.saved_output))
                    dJdm = ufl.dot(adj_input_func, dudm)

            result = self.backend.Function(adj_input_func.function_space())
            result.assign(dJdm)
            return result.vector()

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
