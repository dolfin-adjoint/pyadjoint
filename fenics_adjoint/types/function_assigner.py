import backend
from pyadjoint.tape import get_working_tape
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType, create_overloaded_object
from pyadjoint.enlisting import Enlist


__all__ = ["FunctionAssigner"]


class FunctionAssigner(backend.FunctionAssigner):

    def __init__(self, *args, **kwargs):
        super(FunctionAssigner, self).__init__(*args, **kwargs)
        self.input_spaces = Enlist(args[1])
        self.output_spaces = Enlist(args[0])

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

        block = FunctionAssignerBlock(self, inputs, outputs)
        tape = get_working_tape()
        tape.add_block(block)
        ret = backend.FunctionAssigner.assign(self, outputs.delist(), inputs.delist(), **kwargs)

        if annotate_tape:
            for output in outputs:
                block.add_output(output.block_variable)
        return ret


class FunctionAssignerBlock(Block):
    def __init__(self, assigner, inputs, outputs):
        super(FunctionAssignerBlock, self).__init__()
        for i in inputs:
            self.add_dependency(i)
        self.assigner = assigner

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def prepare_recompute_component(self, inputs, relevant_outputs):
        out_functions = []
        for output in self.get_outputs():
            out_functions.append(backend.Function(output.output.function_space()))
        backend.FunctionAssigner.assign(self.assigner,
                                        self.assigner.output_spaces.delist(out_functions),
                                        self.assigner.input_spaces.delist(inputs))
        return out_functions

    def recompute_component(self, inputs, block_variable, idx, out_functions):
        return out_functions[idx]
