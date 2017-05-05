from .tape import get_working_tape
from .block import Block
from .overloaded_type import OverloadedType


class AdjFloat(OverloadedType, float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, *args)

    def __init__(self, *args, **kwargs):
        super(AdjFloat, self).__init__(*args, **kwargs)
        float.__init__(self, *args, **kwargs)

    def __mul__(self, other):
        output = float.__mul__(self, other)
        if output is NotImplemented:
            return NotImplemented

        block = MulBlock(self, other)

        tape = get_working_tape()
        tape.add_block(block)

        output = AdjFloat(output)
        block.add_output(output.get_block_output())
        
        return output

    def _ad_create_checkpoint(self):
        # Floats are immutable.
        return self

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return self*other

    def _ad_add(self, other):
        return self+other

    def _ad_dot(self, other):
        return float.__mul__(self, other)

# TODO: Not up to date. Low priority for now.
class MulBlock(Block):
    def __init__(self, lfactor, rfactor):
        super(MulBlock, self).__init__()
        self.lfactor = lfactor
        self.rfactor = rfactor

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()

        self.rfactor.add_adj_output(adj_input * self.lfactor)
        self.lfactor.add_adj_output(adj_input * self.rfactor)

