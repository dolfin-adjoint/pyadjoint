import backend
from pyadjoint.tape import get_working_tape
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType, create_overloaded_object
from pyadjoint.enlisting import Enlist


__all__ = []


__ad_functionassigner_assign = backend.FunctionAssigner.assign


def assign(self, *args, **kwargs):
    annotate_tape = kwargs.pop("annotate_tape", True)
    if annotate_tape:
        outputs = Enlist(args[0])
        for i, o in enumerate(outputs):
            if not isinstance(o, OverloadedType):
                outputs[i] = create_overloaded_object(o)

        inputs = Enlist(args[1])
        for j, i in enumerate(outputs):
            if not isinstance(i, OverloadedType):
                inputs[j] = create_overloaded_object(i)

        block = FunctionAssignerBlock(inputs, outputs)
        tape = get_working_tape()
        tape.add_block(block)
    ret = __ad_functionassigner_assign(self, outputs.delist(), inputs.delist(), **kwargs)

    if annotate_tape:
        for output in outputs:
            block.add_output(output.block_variable)
    return ret


backend.FunctionAssigner.assign = assign


class FunctionAssignerBlock(Block):
    def __init__(self, inputs, outputs):
        super(FunctionAssignerBlock, self).__init__()
        for i in inputs:
            self.add_dependency(i)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return None  # Constant._constant_from_values(block_variable.output, inputs[0])
