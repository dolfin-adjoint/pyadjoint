from .tape import get_working_tape, annotate_tape
from .block import Block
from .overloaded_type import OverloadedType, register_overloaded_type


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
            block.add_output(output.block_variable)

        return output

    return annotated_operator


@register_overloaded_type
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
        self.original_block_variable.checkpoint = value

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
        self.terms = [arg.block_variable for arg in args]
        for term in self.terms:
            self.add_dependency(term)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self.operator(*(term.saved_output for term in self.terms))


class PowBlock(FloatOperatorBlock):

    operator = staticmethod(float.__pow__)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        base_value = inputs[0]
        exponent_value = inputs[1]
        adj_input = adj_inputs[0]

        if idx == 0:
            return float.__mul__(float.__mul__(adj_input, exponent_value),
                                 float.__pow__(base_value, exponent_value - 1))
        else:
            from numpy import log
            return float.__mul__(float.__mul__(adj_input, log(base_value)),
                                 float.__pow__(base_value, exponent_value))

class AddBlock(FloatOperatorBlock):

    operator = staticmethod(float.__add__)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_output = 0.
        for term in self.terms:
            tlm_input = term.tlm_value

            if tlm_input is None:
                continue

            tlm_output += tlm_input
        return tlm_output

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]


class SubBlock(FloatOperatorBlock):

    operator = staticmethod(float.__sub__)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if idx == 0:
            return adj_inputs[0]
        else:
            return float.__neg__(adj_inputs[0])


class MulBlock(FloatOperatorBlock):

    operator = staticmethod(float.__mul__)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        other_idx = 0 if idx == 1 else 1
        return float.__mul__(adj_inputs[0], inputs[other_idx])

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_output = 0.
        for i, j in zip((0, 1), (1, 0)):
            tlm_input = self.terms[i].tlm_value

            if tlm_input is None:
                continue

            tlm_output += float.__mul__(tlm_input, self.terms[j].saved_output)
        return tlm_output

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        adj_input = adj_inputs[0]
        hessian_input = hessian_inputs[0]
        other_idx = 0 if idx == 1 else 1
        mixed = 0.0
        for other_idx, bv in relevant_dependencies:
            if other_idx != idx and bv.tlm_value is not None:
                mixed = float.__mul__(adj_input, bv.tlm_value)
        return float.__add__(mixed, float.__mul__(hessian_input, inputs[other_idx]))


class DivBlock(FloatOperatorBlock):

    operator = staticmethod(float.__truediv__)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if idx == 0:
            return float.__mul__(
                adj_inputs[0],
                float.__truediv__(1., inputs[1])
            )
        else:
            return float.__mul__(
                adj_inputs[0],
                float.__neg__(float.__truediv__(
                    inputs[0],
                    float.__pow__(inputs[1], 2)
                ))
            )


class NegBlock(FloatOperatorBlock):

    operator = staticmethod(float.__neg__)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return float.__neg__(adj_inputs[0])

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return float.__neg__(tlm_inputs[0])