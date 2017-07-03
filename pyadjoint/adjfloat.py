from .tape import get_working_tape, annotate_tape
from .block import Block
from .overloaded_type import OverloadedType


class AdjFloat(OverloadedType, float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, *args)

    def __init__(self, *args, **kwargs):
        super(AdjFloat, self).__init__(*args, **kwargs)

    def __mul__(self, other):
        output = float.__mul__(self, other)
        if output is NotImplemented:
            return NotImplemented
        
        if not isinstance(other, OverloadedType):
            other = AdjFloat(other)
            
        output = AdjFloat(output)
        if annotate_tape():
            block = MulBlock(self, other)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.get_block_output())
        
        return output

    def _add(self, other, output):
        if not isinstance(other, OverloadedType):
            other = AdjFloat(other)

        output = AdjFloat(output)
        if annotate_tape():
            block = AddBlock(self, other)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.get_block_output())

        return output

    def __add__(self, other):
        output = float.__add__(self, other)
        if output is NotImplemented:
            return NotImplemented
        return self._add(other, output)

    def __radd__(self, other):
        output = float.__radd__(self, other)
        if output is NotImplemented:
            return NotImplemented
        return self._add(other, output)

    def get_derivative(self, options={}):
        return AdjFloat(self.get_adj_output())

    def adj_update_value(self, value):
        self.original_block_output.checkpoint = value

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


class AddBlock(Block):
    def __init__(self, lterm, rterm):
        super(AddBlock, self).__init__()
        self.add_dependency(lterm.get_block_output())
        self.add_dependency(rterm.get_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()

        self.get_dependencies()[0].add_adj_output(adj_input)
        self.get_dependencies()[1].add_adj_output(adj_input)

    def recompute(self):
        deps = self.get_dependencies()
        self.get_outputs()[0].checkpoint = deps[0].get_saved_output() + deps[1].get_saved_output()


# TODO: Not up to date. Low priority for now.
class MulBlock(Block):
    def __init__(self, lfactor, rfactor):
        super(MulBlock, self).__init__()
        self.add_dependency(lfactor.get_block_output())
        self.add_dependency(rfactor.get_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        deps = self.get_dependencies()

        deps[0].add_adj_output(adj_input * deps[1].get_saved_output())
        deps[1].add_adj_output(adj_input * deps[0].get_saved_output())

    def recompute(self):
        deps = self.get_dependencies()
        self.get_outputs()[0].checkpoint = deps[0].get_saved_output() * deps[1].get_saved_output()

