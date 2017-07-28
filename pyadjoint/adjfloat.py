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
            other = self.__class__(other)

        output = self.__class__(output)
        if annotate_tape():
            block = MulBlock(self, other)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.get_block_output())

        return output

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        output = float.__add__(self, other)
        if output is NotImplemented:
            return NotImplemented

        if not isinstance(other, OverloadedType):
            other = self.__class__(other)

        output = self.__class__(output)
        if annotate_tape():
            block = AddBlock(self, other)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.get_block_output())

        return output

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        output = float.__sub__(self, other)
        if output is NotImplemented:
            return NotImplemented

        if not isinstance(other, OverloadedType):
            other = self.__class__(other)

        output = self.__class__(output)
        if annotate_tape():
            block = SubBlock(self, other)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.get_block_output())

        return output

    def __rsub__(self, other):
        output = float.__sub__(self, other)
        if output is NotImplemented:
            return NotImplemented

        if not isinstance(other, OverloadedType):
            other = self.__class__(other)

        output = self.__class__(output)
        if annotate_tape():
            block = SubBlock(other, self)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.get_block_output())

        return output

    def __pow__(self, power, modulo=None):
        output = float.__pow__(self, power)
        if output is NotImplemented:
            return NotImplemented

        if not isinstance(power, self.__class__):
            power = self.__class__(power)

        output = self.__class__(output)
        if annotate_tape():
            block = PowBlock(self, power)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.get_block_output())

        return output

    def get_derivative(self, options={}):
        return self.__class__(self.get_adj_output())

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


class PowBlock(Block):
    def __init__(self, base, power):
        super(PowBlock, self).__init__()
        self.add_dependency(base.get_block_output())
        self.add_dependency(power.get_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].adj_value

        if adj_input is None:
            return

        dependencies = self.get_dependencies()
        base = dependencies[0]
        exponent = dependencies[1]

        base_value = base.get_saved_output()
        exponent_value = exponent.get_saved_output()

        base_adj = adj_input*exponent_value*base_value**(exponent_value-1)
        base.add_adj_output(base_adj)

        from numpy import log
        exponent_adj = adj_input*base_value**exponent_value*log(base_value)
        exponent.add_adj_output(exponent_adj)

    def recompute(self):
        dependencies = self.get_dependencies()
        base_value = dependencies[0].get_saved_output()
        exponent_value = dependencies[1].get_saved_output()

        new_value = base_value ** exponent_value
        self.get_outputs()[0].checkpoint = new_value


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


class SubBlock(Block):
    def __init__(self, lterm, rterm):
        super(SubBlock, self).__init__()
        self.add_dependency(lterm.get_block_output())
        self.add_dependency(rterm.get_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()

        self.get_dependencies()[0].add_adj_output(adj_input)
        self.get_dependencies()[1].add_adj_output(adj_input)

    def recompute(self):
        deps = self.get_dependencies()
        self.get_outputs()[0].checkpoint = deps[0].get_saved_output() - deps[1].get_saved_output()


# TODO: Not up to date. Low priority for now.
class MulBlock(Block):
    def __init__(self, lfactor, rfactor):
        super(MulBlock, self).__init__()
        self.add_dependency(lfactor.get_block_output())
        self.add_dependency(rfactor.get_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        deps = self.get_dependencies()

        deps[0].add_adj_output(float.__mul__(adj_input, deps[1].get_saved_output()))
        deps[1].add_adj_output(float.__mul__(adj_input, deps[0].get_saved_output()))

    def recompute(self):
        deps = self.get_dependencies()
        self.get_outputs()[0].checkpoint = deps[0].get_saved_output() * deps[1].get_saved_output()
