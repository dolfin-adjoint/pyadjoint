from pyadjoint import Block
import numpy

from pyadjoint.reduced_functional_numpy import gather


def constant_from_values(constant, values=None):
    """Returns a new Constant with `constant.values()` while preserving `constant.ufl_shape`.

    If the optional argument `values` is provided, then `values` will be the values of the
    new Constant instead, while still preserving the ufl_shape of `constant`.

    Args:
        constant: A constant with the ufl_shape to preserve.
        values (numpy.array): An optional argument to use instead of constant.values().

    Returns:
        Constant: The created Constant of the same type as `constant`.

    """
    values = constant.values() if values is None else values
    return type(constant)(numpy.reshape(values, constant.ufl_shape))


class ConstantAssignBlock(Block):
    def __init__(self, other):
        super(ConstantAssignBlock, self).__init__()
        self.add_dependency(other)
        self.assigned_float = isinstance(other, float)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_output = adj_inputs[0]
        if self.assigned_float:
            # Convert to float
            adj_output = gather(adj_output)
            adj_output = float(adj_output[0])
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return constant_from_values(block_variable.output, tlm_inputs[0])

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return constant_from_values(block_variable.output, inputs[0])
