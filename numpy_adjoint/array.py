import numpy

from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType, register_overloaded_type, create_overloaded_object
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from .utils import adjoint_broadcast


def annotate_operator(operator):
    """Decorate numpy operator like __add__, __sub__, etc.

    The provided operator is only expected to create the Block that
    corresponds to this operation. The decorator returns a wrapper code
    that checks whether annotation is needed, ensures all arguments are
    overloaded, calls the block-creating operator, and puts the Block
    on tape."""

    # the actual numpy operation is derived from the name of operator
    op = getattr(numpy.ndarray, operator.__name__)

    def annotated_operator(self, other):
        annotate = annotate_tape()
        if annotate:
            other = create_overloaded_object(other)
            block = operator(self, other)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            out = op(self, other)
        out = create_overloaded_object(out)

        if annotate:
            block.add_output(out.create_block_variable())

        return out

    return annotated_operator


def annotate_inplace_operator(operator):
    # the actual numpy operation is derived from the name of operator
    op = getattr(numpy.ndarray, operator.__name__)

    def annotated_operator(self, other):
        annotate = annotate_tape()
        if annotate:
            other = create_overloaded_object(other)
            block = operator(self, other)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            out = op(self, other)

        if annotate:
            block.add_output(self.create_block_variable())

        return out

    return annotated_operator


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

    def adj_update_value(self, value):
        self[:] = value

    def __getitem__(self, item):
        annotate = annotate_tape()
        if annotate:
            block = SliceBlock(self, item)
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

    def _ad_dot(self, other):
        return self.dot(other)

    def _ad_mul(self, other):
        return other * self

    def _ad_add(self, other):
        return self + other

    @annotate_operator
    def __add__(self, other):
        return AddBlock(self, other)

    @annotate_operator
    def __mul__(self, other):
        return MulBlock(self, other)

    @annotate_operator
    def __pow__(self, power, modulo=None):
        return PowBlock(self, power, modulo)

    @annotate_operator
    def __truediv__(self, other):
        return DivBlock(self, other)

    @annotate_operator
    def __sub__(self, other):
        return SubBlock(self, other)

    @no_annotations
    def __str__(self):
        return super().__str__()

    @annotate_inplace_operator
    def __imul__(self, other):
        return MulBlock(self, other)


def array(*args, **kwargs):
    arr = numpy.array(*args, **kwargs)
    return create_overloaded_object(arr)


def zeros(*args, **kwargs):
    arr = numpy.zeros(*args, **kwargs)
    return create_overloaded_object(arr)


class SliceBlock(Block):
    def __init__(self, array, item):
        super().__init__()
        self.add_dependency(array)
        self.item = item

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_output = numpy.zeros(inputs[0].shape)
        adj_output[self.item] = adj_inputs[0]
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0][self.item]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0][self.item]


class AddBlock(Block):
    def __init__(self, arr1, arr2):
        super().__init__()
        self.add_dependency(arr1)
        self.add_dependency(arr2)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0] + tlm_inputs[1]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0] + inputs[1]


class MulBlock(Block):
    def __init__(self, arr1, arr2):
        super().__init__()
        self.add_dependency(arr1)
        self.add_dependency(arr2)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        other_idx = 0 if idx == 1 else 1
        adj_output = inputs[other_idx] * adj_inputs[0]
        adj_output = adjoint_broadcast(adj_output, numpy.array(inputs[idx]).shape)
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_output = 0.
        if tlm_inputs[0] is not None:
            tlm_output += tlm_inputs[0] * inputs[1]
        if tlm_inputs[1] is not None:
            tlm_output += tlm_inputs[1] * inputs[0]
        return tlm_output

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        hessian_output = self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)
        mixed = 0.
        for other_idx, bv in relevant_dependencies:
            if other_idx != idx and bv.tlm_value is not None:
                mixed = adj_inputs[0] * bv.tlm_value
        mixed = adjoint_broadcast(mixed, numpy.array(inputs[idx]).shape)
        return hessian_output + mixed

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0] * inputs[1]


class DivBlock(Block):
    def __init__(self, arr1, arr2):
        super().__init__()
        self.add_dependency(arr1)
        self.add_dependency(arr2)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if idx == 0:
            return adj_inputs[0] / inputs[1]
        else:
            return adj_inputs[0] * (-inputs[0] / inputs[1]**2)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_output = 0.

        if tlm_inputs[0] is not None:
            tlm_output += tlm_inputs[0] / inputs[1]
        if tlm_inputs[1] is not None:
            tlm_output += tlm_inputs[1] * (-inputs[0] / inputs[1]**2)
        return tlm_output

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        adj_input = adj_inputs[0]
        denominator_value = inputs[1]
        return -adj_input / denominator_value**2

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        numerator = inputs[0]
        denominator = inputs[1]
        adj_input = adj_inputs[0]
        mixed = prepared

        hessian_output = self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx)

        if idx == 1 and block_variable.tlm_value is not None:
            # d^2f/dx^2 where x is denominator
            hessian_output += adj_input * 2. * numerator / denominator ** 3 * block_variable.tlm_value

        for other_idx, bv in relevant_dependencies:
            if other_idx != idx and bv.tlm_value is not None:
                hessian_output += mixed * bv.tlm_value
        return hessian_output

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0] / inputs[1]


class PowBlock(Block):
    def __init__(self, arr, power, modulo):
        super().__init__()
        self.add_dependency(arr)
        self.add_dependency(power)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        base_value = inputs[0]
        exponent_value = inputs[1]
        adj_input = adj_inputs[0]

        if idx == 0:
            return adj_input * exponent_value * base_value**(exponent_value - 1)
        else:
            return adj_input * numpy.log(base_value) * base_value**exponent_value

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        base_value = inputs[0]
        exponent_value = inputs[1]
        tlm_base = tlm_inputs[0]
        tlm_exponent = tlm_inputs[1]

        dfdb = 0.
        dfde = 0.
        if tlm_base is not None:
            dfdb = tlm_base * exponent_value * base_value ** (exponent_value - 1)

        if tlm_exponent is not None:
            dfde = tlm_exponent * numpy.log(base_value) * base_value ** exponent_value

        return dfdb + dfde

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        base_value = inputs[0]
        exponent_value = inputs[1]
        adj_input = adj_inputs[0]
        mixed_derivatives = (adj_input * base_value ** (exponent_value - 1)
                             * (exponent_value * numpy.log(base_value) + 1))
        return mixed_derivatives

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        base_value = inputs[0]
        exponent_value = inputs[1]
        adj_input = adj_inputs[0]
        mixed_derivatives = prepared

        hessian_output = self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx)
        for other_idx, bv in relevant_dependencies:
            if other_idx == idx and bv.tlm_value is not None:
                # Compute adj_value * d^2 f / dx^2 tlm_value
                if idx == 0:
                    # x is base
                    hessian_output += (adj_input * exponent_value * (exponent_value - 1)
                                       * base_value ** (exponent_value - 2) * bv.tlm_value)
                else:
                    # x is exponent
                    hessian_output += adj_input * numpy.log(
                        base_value) ** 2 * base_value ** exponent_value * bv.tlm_value
            else:
                # Mixed derivatives
                if bv.tlm_value is not None:
                    hessian_output += bv.tlm_value * mixed_derivatives
        return hessian_output

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0]**inputs[1]


class SubBlock(Block):
    def __init__(self, arr1, arr2):
        super().__init__()
        self.add_dependency(arr1)
        self.add_dependency(arr2)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        sign = -1 if idx == 1 else 1
        return sign * adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_output = 0.
        if tlm_inputs[0] is not None:
            tlm_output += tlm_inputs[0]
        if tlm_inputs[1] is not None:
            tlm_output -= tlm_inputs[1]
        return tlm_output

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return inputs[0] - inputs[1]
