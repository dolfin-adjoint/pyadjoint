import backend
from pyadjoint.tape import get_working_tape, annotate_tape
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType, create_overloaded_object
from pyadjoint.enlisting import Enlist


__all__ = ["FunctionAssigner"]


class FunctionAssigner(backend.FunctionAssigner):

    def __init__(self, *args, **kwargs):
        super(FunctionAssigner, self).__init__(*args, **kwargs)
        self.input_spaces = Enlist(args[1])
        self.output_spaces = Enlist(args[0])
        self.adj_assigner = backend.FunctionAssigner(args[1],
                                                     args[0],
                                                     **kwargs)

    def assign(self, *args, **kwargs):
        annotate = annotate_tape(kwargs)
        if annotate:
            outputs = Enlist(args[0])
        for i, o in enumerate(outputs):
            if not isinstance(o, OverloadedType):
                outputs[i] = create_overloaded_object(o)

        inputs = Enlist(args[1])
        for j, i in enumerate(outputs):
            if not isinstance(i, OverloadedType):
                inputs[j] = create_overloaded_object(i)

        block = FunctionAssignerBlock(self, inputs)
        tape = get_working_tape()
        tape.add_block(block)
        ret = backend.FunctionAssigner.assign(self, outputs.delist(), inputs.delist(), **kwargs)

        if annotate:
            for output in outputs:
                block.add_output(output.block_variable)
        return ret


class FunctionAssignerBlock(Block):
    def __init__(self, assigner, inputs):
        super(FunctionAssignerBlock, self).__init__()
        for i in inputs:
            self.add_dependency(i)
        self.assigner = assigner

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        adj_assigner = self.assigner.adj_assigner
        inp_functions = []
        for i in range(len(adj_inputs)):
            f_in = backend.Function(self.assigner.output_spaces[i])
            if adj_inputs[i] is not None:
                f_in.vector()[:] = adj_inputs[i]
            inp_functions.append(f_in)
        out_functions = []
        for j in range(len(self.assigner.input_spaces)):
            f_out = backend.Function(self.assigner.input_spaces[j])
            out_functions.append(f_out)
        adj_assigner.assign(self.assigner.input_spaces.delist(out_functions),
                            self.assigner.output_spaces.delist(inp_functions))
        return out_functions

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return prepared[idx].vector()

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self.prepare_recompute_component(tlm_inputs, relevant_outputs)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return prepared[idx]

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, hessian_inputs, relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs,
                                           block_variable, idx, prepared)

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
