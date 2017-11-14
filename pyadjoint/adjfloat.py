from .tape import get_working_tape, annotate_tape
from .block import Block
from .overloaded_type import OverloadedType


def annotate_operator(operator):
    """Decorate float operator like __add__, __sub__, etc.

    The provided operator is only expected to create the Block that
    corresponds to this operation. The decorator returns a wrapper code
    that checks whether annotation is needed, ensures all arguments are
    overloaded, calls the block-creating operator, and puts the Block
    on tape."""

    # the actual float operation is derived from the name of operator
    try:
        float_op = getattr(float, operator.__name__)
    except AttributeError:
        if operator.__name__ == '__div__':
            float_op = float.__truediv__
        else:
            raise

    def annotated_operator(self, *args):
        output = float_op(self, *args)
        if output is NotImplemented:
            return NotImplemented

        # ensure all arguments are of OverloadedType
        args = [arg if isinstance(arg, OverloadedType) else self.__class__(arg) for arg in args]

        output = self.__class__(output)
        if annotate_tape():
            block = operator(self, *args)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.get_block_output())

        return output

    return annotated_operator


class AdjFloat(OverloadedType, float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, *args)

    def __init__(self, *args, **kwargs):
        super(AdjFloat, self).__init__(*args, **kwargs)

    @annotate_operator
    def __mul__(self, other):
        return MulBlock(self, other)

    @annotate_operator
    def __div__(self, other):
        return DivBlock(self, other)

    @annotate_operator
    def __truediv__(self, other):
        return DivBlock(self, other)

    @annotate_operator
    def __neg__(self):
        return NegBlock(self)

    @annotate_operator
    def __rmul__(self, other):
        return MulBlock(self, other)

    @annotate_operator
    def __add__(self, other):
        return AddBlock(self, other)

    @annotate_operator
    def __radd__(self, other):
        return AddBlock(self, other)

    @annotate_operator
    def __sub__(self, other):
        return SubBlock(self, other)

    @annotate_operator
    def __rsub__(self, other):
        # NOTE: order is important here
        return SubBlock(other, self)

    @annotate_operator
    def __pow__(self, power):
        return PowBlock(self, power)

    def _ad_convert_type(self, value, options={}):
        return AdjFloat(value)

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

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        dst = float(src[offset:offset+1])
        offset += 1
        return dst, offset

    @staticmethod
    def _ad_to_list(value):
        return [value]

    def _ad_copy(self):
        return self


class FloatOperatorBlock(Block):

    # the float operator annotated in this Block
    operator = None

    def __init__(self, *args):
        super(FloatOperatorBlock, self).__init__()
        # the terms are stored seperately here and added as dependencies
        # this is because get_dependencies() only returns the terms with
        # duplicates taken out; for evaluation however order and position
        # of the terms is significant
        self.terms = [arg.get_block_output() for arg in args]
        for term in self.terms:
            self.add_dependency(term)

    def recompute(self):
        self.get_outputs()[0].checkpoint = self.operator(*(term.get_saved_output() for term in self.terms))


class PowBlock(FloatOperatorBlock):

    operator = staticmethod(float.__pow__)

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return

        base = self.terms[0]
        exponent = self.terms[1]

        base_value = base.get_saved_output()
        exponent_value = exponent.get_saved_output()

        base_adj = float.__mul__(float.__mul__(adj_input, exponent_value),
                                 float.__pow__(base_value, exponent_value-1))
        base.add_adj_output(base_adj)

        from numpy import log
        exponent_adj = float.__mul__(float.__mul__(adj_input, log(base_value)),
                                     float.__pow__(base_value, exponent_value))
        exponent.add_adj_output(exponent_adj)


class AddBlock(FloatOperatorBlock):

    operator = staticmethod(float.__add__)

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        if adj_input is None:
            return

        for term in self.terms:
            term.add_adj_output(adj_input)


class SubBlock(FloatOperatorBlock):

    operator = staticmethod(float.__sub__)

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        if adj_input is None:
            return

        self.terms[0].add_adj_output(adj_input)
        self.terms[1].add_adj_output(float.__neg__(adj_input))


class MulBlock(FloatOperatorBlock):

    operator = staticmethod(float.__mul__)

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        if adj_input is None:
            return

        self.terms[0].add_adj_output(float.__mul__(adj_input, self.terms[1].get_saved_output()))
        self.terms[1].add_adj_output(float.__mul__(adj_input, self.terms[0].get_saved_output()))


class DivBlock(FloatOperatorBlock):

    operator = staticmethod(float.__truediv__)

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        if adj_input is None:
            return

        self.terms[0].add_adj_output(float.__mul__(
            adj_input,
            float.__truediv__(1., self.terms[1].get_saved_output())
        ))
        self.terms[1].add_adj_output(float.__mul__(
            adj_input,
            float.__neg__(float.__truediv__(
                self.terms[0].get_saved_output(),
                float.__pow__(self.terms[1].get_saved_output(), 2)
            ))
        ))


class NegBlock(FloatOperatorBlock):

    operator = staticmethod(float.__neg__)

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        if adj_input is None:
            return

        self.terms[0].add_adj_output(float.__neg__(adj_input))
