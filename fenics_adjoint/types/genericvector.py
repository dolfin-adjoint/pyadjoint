import backend

from pyadjoint.tape import stop_annotating, annotate_tape, get_working_tape
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType, register_overloaded_type


__all__ = []


class GenericVector(OverloadedType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _ad_init_object(cls, obj):
        obj.__class__ = cls
        OverloadedType.__init__(obj)
        return obj

    def _ad_create_checkpoint(self):
        return self.copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_name(self):
        return str(self)

    def get_local(self, *args, **kwargs):
        annotate = annotate_tape(kwargs)
        if annotate:
            block = VectorGetLocalBlock(self)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            out = backend.GenericVector.get_local(self, *args, **kwargs)
        out = create_overloaded_object(out)

        if annotate:
            block.add_output(out.create_block_variable())

        return out

    def __getitem__(self, item):
        annotate = annotate_tape()
        if annotate:
            block = SliceBlock(self, item)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            out = backend.GenericVector.__getitem__(self, item)
        out = create_overloaded_object(out)

        if annotate:
            block.add_output(out.create_block_variable())

        return out

    def __setitem__(self, key, value):
        annotate = annotate_tape()
        if annotate:
            value = create_overloaded_object(value)
            block = SetSliceBlock(self, key, value)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            out = backend.GenericVector.__setitem__(self, key, value)

        if annotate:
            block.add_output(self.create_block_variable())

        return out


class PETScVector(GenericVector, backend.PETScVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        backend.PETScVector.__init__(self, *args, **kwargs)


register_overloaded_type(PETScVector, backend.PETScVector)


class SliceBlock(Block):
    def __init__(self, vec, item):
        super().__init__()
        self.add_dependency(vec)
        self.item = item

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_vec = inputs[0].copy()
        adj_vec.zero()
        adj_vec[self.item] = adj_inputs[0]
        return adj_vec

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0][self.item]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0][self.item]


class VectorGetLocalBlock(Block):
    def __init__(self, vec):
        super().__init__()
        self.add_dependency(vec)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_output = inputs[0].copy()
        adj_output[:] = adj_inputs[0]
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0].get_local()

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0].get_local()


class SetSliceBlock(Block):
    def __init__(self, arr, item, value):
        super().__init__()
        self.add_dependency(arr)
        self.add_dependency(value)
        self.item = item

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        if idx == 0:
            # Derivative wrt to vector
            adj_output = adj_input.copy()
            adj_output[self.item] = 0.
        else:
            inp = inputs[1]
            if isinstance(inp, GenericVector):
                # Assume always assigned full slice
                adj_output = adj_input.copy()
            elif isinstance(inp, float):

                if isinstance(self.item, slice):
                    adj_output = adj_input[self.item].sum()
                else:
                    adj_output = adj_input[self.item]
            else:
                # Assume numpy array
                adj_output = adj_input[self.item]
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if tlm_inputs[0] is not None:
            tlm_input_0 = tlm_inputs[0].copy()
        else:
            vec = inputs[0]
            tlm_input_0 = type(vec)(vec.mpi_comm(), vec.size())

        tlm_input_1 = 0.
        if tlm_inputs[1] is not None:
            tlm_input_1 = tlm_inputs[1]

        tlm_input_0[self.item] = tlm_input_1
        return tlm_input_0

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        inputs[0][self.item] = inputs[1]
        return inputs[0]
