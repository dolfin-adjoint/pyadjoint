from .block import Block
from .overloaded_type import OverloadedType, register_overloaded_type, create_overloaded_object
from .tape import get_working_tape, annotate_tape, stop_annotating


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

    def _ad_create_checkpoint(self):
        # Floats are immutable.
        return self

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return self * other

    def _ad_add(self, other):
        return self + other

    def _ad_dot(self, other):
        return float.__mul__(self, other)

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        dst = type(dst)(src[offset:offset + 1])
        offset += 1
        return dst, offset

    @staticmethod
    def _ad_to_list(value):
        return [value]

    def _ad_copy(self):
        return self


_min = min
_max = max


def min(a, b, **kwargs):
    annotate = annotate_tape(kwargs)
    if annotate:
        # Ensure a and b are of OverloadedType
        a = create_overloaded_object(a)
        b = create_overloaded_object(b)

        block = MinBlock(a, b)
        tape = get_working_tape()
        tape.add_block(block)

    with stop_annotating():
        out = _min(a, b)
    out = AdjFloat(out)

    if annotate:
        block.add_output(out.block_variable)
    return out


def max(a, b, **kwargs):
    annotate = annotate_tape(kwargs)
    if annotate:
        # Ensure a and b are of OverloadedType
        a = create_overloaded_object(a)
        b = create_overloaded_object(b)

        block = MaxBlock(a, b)
        tape = get_working_tape()
        tape.add_block(block)

    with stop_annotating():
        out = _max(a, b)
    out = AdjFloat(out)

    if annotate:
        block.add_output(out.block_variable)
    return out


class MinBlock(Block):
    def __init__(self, a, b):
        super().__init__()
        self.add_dependency(a)
        self.add_dependency(b)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        active_idx = 0 if inputs[0] <= inputs[1] else 1
        if idx == active_idx:
            return adj_input
        else:
            return 0.

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        idx = 0 if inputs[0] <= inputs[1] else 1
        return tlm_inputs[idx]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return _min(inputs[0], inputs[1])


class MaxBlock(Block):
    def __init__(self, a, b):
        super().__init__()
        self.add_dependency(a)
        self.add_dependency(b)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        active_idx = 0 if inputs[0] >= inputs[1] else 1
        if idx == active_idx:
            return adj_input
        else:
            return 0.

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        idx = 0 if inputs[0] >= inputs[1] else 1
        return tlm_inputs[idx]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        r = _max(inputs[0], inputs[1])
        return r


class FloatOperatorBlock(Block):
    # the float operator annotated in this Block
    operator = None
    symbol = None

    def __init__(self, *args):
        super(FloatOperatorBlock, self).__init__()
        # the terms are stored seperately here and added as dependencies
        # this is because get_dependencies() only returns the terms with
        # duplicates taken out; for evaluation however order and position
        # of the terms is significant
        self.terms = [arg.block_variable for arg in args]
        for dep in args:
            self.add_dependency(dep)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self.operator(*(term.saved_output for term in self.terms))

    def __str__(self):
        return f"{self.terms[0]} {self.symbol} {self.terms[1]}"


class PowBlock(FloatOperatorBlock):
    operator = staticmethod(float.__pow__)
    symbol = "**"

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

    def evaluate_tlm(self, markings=False):
        output = self.get_outputs()[0]

        base = self.terms[0]
        exponent = self.terms[1]

        base_value = base.saved_output
        exponent_value = exponent.saved_output

        if base.tlm_value is not None:
            base_tlm = float.__mul__(float.__mul__(base.tlm_value, exponent_value),
                                     float.__pow__(base_value, exponent_value - 1))
            output.add_tlm_output(base_tlm)

        if exponent.tlm_value is not None:
            from numpy import log
            exponent_adj = float.__mul__(float.__mul__(exponent.tlm_value, log(base_value)),
                                         float.__pow__(base_value, exponent_value))
            output.add_tlm_output(exponent_adj)

    def evaluate_hessian(self, markings=False):
        output = self.get_outputs()[0]
        hessian_input = output.hessian_value
        adj_input = output.adj_value
        if hessian_input is None:
            return

        base = self.terms[0]
        exponent = self.terms[1]

        base_value = base.saved_output
        exponent_value = exponent.saved_output

        # First we do the base hessian (minus the mixed derivative)
        if base.tlm_value is not None:
            second_order = float.__mul__(float.__mul__(
                float.__mul__(adj_input, float.__mul__(exponent_value, exponent_value - 1)),
                float.__pow__(base_value, exponent_value - 2)), base.tlm_value)
            base.add_hessian_output(second_order)

        first_order = float.__mul__(float.__mul__(hessian_input, exponent_value),
                                    float.__pow__(base_value, exponent_value - 1))
        base.add_hessian_output(first_order)

        # Then we do the exponent hessian (minus the mixed derivative)
        from numpy import log
        if exponent.tlm_value is not None:
            second_order = float.__mul__(float.__mul__(float.__mul__(adj_input, float.__pow__(log(base_value), 2)),
                                                       float.__pow__(base_value, exponent_value)), exponent.tlm_value)
            exponent.add_hessian_output(second_order)

        first_order = float.__mul__(float.__mul__(hessian_input, log(base_value)),
                                    float.__pow__(base_value, exponent_value))
        exponent.add_hessian_output(first_order)

        # Lastly we add mixed derivative terms
        mixed = float.__mul__(adj_input, float.__mul__(
            float.__pow__(base_value, exponent_value - 1),
            float.__add__(float.__mul__(exponent_value, log(base_value)), 1)))
        if exponent.tlm_value is not None:
            base.add_hessian_output(float.__mul__(exponent.tlm_value, mixed))
        if base.tlm_value is not None:
            exponent.add_hessian_output(float.__mul__(base.tlm_value, mixed))


class AddBlock(FloatOperatorBlock):
    operator = staticmethod(float.__add__)
    symbol = "+"

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
    symbol = "-"

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if idx == 0:
            return adj_inputs[0]
        else:
            return float.__neg__(adj_inputs[0])

    def evaluate_tlm(self, markings=False):
        output = self.get_outputs()[0]
        tlm_input_0 = self.terms[0].tlm_value
        if tlm_input_0 is not None:
            output.add_tlm_output(tlm_input_0)
        tlm_input_1 = self.terms[1].tlm_value
        if tlm_input_1 is not None:
            output.add_tlm_output(float.__neg__(tlm_input_1))

    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return

        self.terms[0].add_hessian_output(hessian_input)
        self.terms[1].add_hessian_output(float.__neg__(hessian_input))


class MulBlock(FloatOperatorBlock):
    operator = staticmethod(float.__mul__)
    symbol = "*"

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
    symbol = "/"

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

    def evaluate_tlm(self, markings=False):
        output = self.get_outputs()[0]

        if self.terms[0].tlm_value is not None:
            output.add_tlm_output(float.__mul__(
                self.terms[0].tlm_value,
                float.__truediv__(1., self.terms[1].saved_output)
            ))
        if self.terms[1].tlm_value is not None:
            output.add_tlm_output(float.__mul__(
                self.terms[1].tlm_value,
                float.__neg__(float.__truediv__(
                    self.terms[0].saved_output,
                    float.__pow__(self.terms[1].saved_output, 2)
                ))
            ))

    def evaluate_hessian(self, markings=False):
        output = self.get_outputs()[0]
        hessian_input = output.hessian_value
        adj_input = output.adj_value
        if hessian_input is None:
            return

        numerator = self.terms[0]
        denominator = self.terms[1]

        numerator_value = numerator.saved_output
        denominator_value = denominator.saved_output

        # The function is linear in the numerator
        numerator.add_hessian_output(float.__mul__(
            hessian_input,
            float.__truediv__(1., denominator_value)
        ))

        # Now for the denominator
        denominator.add_hessian_output(float.__mul__(
            hessian_input,
            float.__neg__(float.__truediv__(
                numerator_value,
                float.__pow__(denominator_value, 2.)
            ))
        ))

        if denominator.tlm_value is not None:
            denominator.add_hessian_output(float.__mul__(
                float.__mul__(
                    adj_input,
                    float.__truediv__(
                        float.__mul__(2., numerator_value),
                        float.__pow__(denominator_value, 3)
                    )
                ), denominator.tlm_value))

        # Now for mixed derivative
        mixed = float.__neg__(float.__truediv__(adj_input, float.__pow__(denominator_value, 2)))
        if denominator.tlm_value is not None:
            numerator.add_hessian_output(float.__mul__(denominator.tlm_value, mixed))
        if numerator.tlm_value is not None:
            denominator.add_hessian_output(float.__mul__(numerator.tlm_value, mixed))


class NegBlock(FloatOperatorBlock):
    operator = staticmethod(float.__neg__)
    symbol = "-"

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return float.__neg__(adj_inputs[0])

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return float.__neg__(tlm_inputs[0])

    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return

        self.terms[0].add_hessian_output(float.__neg__(hessian_input))

    def __str__(self):
        return f"{self.symbol} {self.terms[0]}"
