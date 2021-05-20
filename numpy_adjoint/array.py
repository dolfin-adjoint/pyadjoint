import numpy
from pyadjoint.overloaded_type import OverloadedType, register_overloaded_type, create_overloaded_object
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape
from pyadjoint.block import Block


@register_overloaded_type
class ndarray(OverloadedType, numpy.ndarray):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def _ad_init_object(cls, obj):
        return cls(obj.shape, numpy.float_, buffer=obj)

    def _ad_create_checkpoint(self):
        return self.copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def __getitem__(self, item):
        annotate = annotate_tape()
        if annotate:
            block = NumpyArraySliceBlock(self, item)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            out = numpy.ndarray.__getitem__(self, item)

        if annotate:
            out = create_overloaded_object(out)
            block.add_output(out.create_block_variable())
        return out

    def _ad_convert_type(self, value, options={}):
        return value

    def __array_finalize__(self, obj):
        OverloadedType.__init__(self)


class NumpyArraySliceBlock(Block):
    def __init__(self, array, item):
        super().__init__()
        self.add_dependency(array)
        self.item = item

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_output = numpy.zeros(inputs[0].shape)
        adj_output[self.item] = adj_inputs[0]
        return adj_output

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0][self.item]
