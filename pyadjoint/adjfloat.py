from functools import cached_property, lru_cache, wraps
from itertools import count
import numbers
import operator
from .block import Block
from .overloaded_type import OverloadedType, register_overloaded_type
from .tape import get_working_tape, annotate_tape
import numpy as np
import sympy as sp

_op_fns = {}


def register_function(np_operator):
    def register(cls):
        _op_fns[cls.__name__] = np_operator
        return cls
    return register


@register_function(np.hypot)
class _pyadjoint_hypot(sp.Function):
    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0] / _pyadjoint_hypot(self.args[0],
                                                   self.args[1])
        elif argindex == 2:
            return self.args[1] / _pyadjoint_hypot(self.args[0],
                                                   self.args[1])


@register_function(np.expm1)
class _pyadjoint_expm1(sp.Function):
    def fdiff(self, argindex=1):
        if argindex == 1:
            return sp.exp(self.args[0])


@register_function(np.log1p)
class _pyadjoint_log1p(sp.Function):
    def fdiff(self, argindex=1):
        if argindex == 1:
            return sp.Integer(1) / (sp.Integer(1) + self.args[0])


@lru_cache(maxsize=128)
def codegen(expr, symbols, diff=()):
    for idx in diff:
        expr = expr.diff(symbols[idx])
    return sp.lambdify(symbols, expr, modules=["numpy", _op_fns])


class Operator:
    _symbol_count = count()

    def __init__(self, fn, nargs):
        self._fn = fn
        self._nargs = nargs

    @property
    def nargs(self):
        return self._nargs

    @cached_property
    def symbols(self):
        return tuple(sp.Symbol(f"_pyadjoint_symbol_{next(self._symbol_count)}", real=True) for _ in range(self.nargs))

    @cached_property
    def expr(self):
        return self._fn(*self.symbols)

    def codegen(self, diff=()):
        return codegen(self.expr, self.symbols, diff=diff)


class AdjFloatExprBlock(Block):
    def __init__(self, sp_operator, *args):
        super().__init__()
        self._sp_operator = sp_operator
        for arg in args:
            self.add_dependency(arg)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input, = adj_inputs
        return self._sp_operator.codegen(diff=(idx,))(*inputs) * adj_input

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if idx != 0:
            raise ValueError("Unexpected idx")
        val = 0.0
        for idx1 in range(self._sp_operator.nargs):
            if tlm_inputs[idx1] is not None:
                val += self._sp_operator.codegen(diff=(idx1,))(*inputs) * tlm_inputs[idx1]
        return val

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        hessian_input, = hessian_inputs
        adj_input, = adj_inputs
        tlm_inputs = tuple(dep.tlm_value for dep in self.get_dependencies())
        val = 0.0
        if hessian_input is not None:
            val += self._sp_operator.codegen(diff=(idx,))(*inputs) * hessian_input
        for idx1 in range(self._sp_operator.nargs):
            if adj_input is not None and tlm_inputs[idx1] is not None:
                val += self._sp_operator.codegen(diff=(idx, idx1))(*inputs) * adj_input * tlm_inputs[idx1]
        return val

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self._sp_operator.codegen(diff=())(*inputs)


def annotate_operator(sp_operator):
    def wrapper(fn):
        @wraps(fn)
        def annotated_operator(*args):
            output = fn(*args)
            if not isinstance(output, numbers.Complex):
                # Not annotated
                return output
            output = AdjFloat(output)  # Error here if not real

            if annotate_tape():
                args = list(args)
                adjfloat_bv = set()
                for i, arg in enumerate(args):
                    if isinstance(arg, OverloadedType):
                        pass
                    elif isinstance(arg, numbers.Complex):
                        args[i] = AdjFloat(arg)  # Error here if not real
                    else:
                        # Not annotated
                        return output

                block = AdjFloatExprBlock(sp_operator, *args)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(output.block_variable)
            return output
        return annotated_operator
    return wrapper


def roperator(operator):
    def roperator(a, b):
        return operator(b, a)
    return roperator


_ops = {}


def register_operator(np_operator, sp_operator, nargs):
    def wrapped_operator(*args):
        if len(args) != nargs:
            return NotImplemented
        return np_operator(*(float(arg) if isinstance(arg, AdjFloat) else arg for arg in args))
    _ops[np_operator] = annotate_operator(Operator(sp_operator, nargs))(wrapped_operator)
    return _ops[np_operator]


@register_overloaded_type
class AdjFloat(OverloadedType, float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, *args)

    def __init__(self, *args, **kwargs):
        super(AdjFloat, self).__init__(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        if len(kwargs) > 0:
            return NotImplemented
        if len(inputs) == 0:
            return NotImplemented
        if ufunc in _ops:
            return _ops[ufunc](*inputs)
        else:
            # Not annotated
            return ufunc(*(float(arg) if isinstance(arg, AdjFloat) else arg for arg in inputs))

    @annotate_operator(Operator(sp.Abs, 1))
    def __abs__(self):
        return super().__abs__()

    @annotate_operator(Operator(operator.pos, 1))
    def __pos__(self):
        return super().__pos__()

    @annotate_operator(Operator(operator.neg, 1))
    def __neg__(self):
        return super().__neg__()

    @annotate_operator(Operator(operator.mul, 2))
    def __mul__(self, other):
        return super().__mul__(other)

    @annotate_operator(Operator(roperator(operator.mul), 2))
    def __rmul__(self, other):
        return super().__rmul__(other)

    @annotate_operator(Operator(operator.truediv, 2))
    def __truediv__(self, other):
        return super().__truediv__(other)

    @annotate_operator(Operator(roperator(operator.truediv), 2))
    def __rtruediv__(self, other):
        return super().__rtruediv__(other)

    @annotate_operator(Operator(operator.add, 2))
    def __add__(self, other):
        return super().__add__(other)

    @annotate_operator(Operator(roperator(operator.add), 2))
    def __radd__(self, other):
        return super().__radd__(other)

    @annotate_operator(Operator(operator.sub, 2))
    def __sub__(self, other):
        return super().__sub__(other)

    @annotate_operator(Operator(roperator(operator.sub), 2))
    def __rsub__(self, other):
        return super().__rsub__(other)

    @annotate_operator(Operator(operator.pow, 2))
    def __pow__(self, other):
        return super().__pow__(other)

    @annotate_operator(Operator(roperator(operator.pow), 2))
    def __rpow__(self, other):
        return super().__rpow__(other)

    abs = register_operator(np.abs, operator.abs, 1)
    positive = register_operator(np.positive, operator.pos, 1)
    negative = register_operator(np.negative, operator.neg, 1)
    multiply = register_operator(np.multiply, operator.mul, 2)
    divide = register_operator(np.divide, operator.truediv, 2)
    add = register_operator(np.add, operator.add, 2)
    subtract = register_operator(np.subtract, operator.sub, 2)
    power = register_operator(np.power, operator.pow, 2)
    minimum = register_operator(
        np.minimum,
        lambda self, other: sp.Piecewise((self, self <= other),
                                         (other, True)),
        2)
    maximum = register_operator(
        np.maximum,
        lambda self, other: sp.Piecewise((self, self >= other),
                                         (other, True)),
        2)

    sin = register_operator(np.sin, sp.sin, 1)
    cos = register_operator(np.cos, sp.cos, 1)
    tan = register_operator(np.tan, sp.tan, 1)
    arcsin = register_operator(np.arcsin, sp.asin, 1)
    arccos = register_operator(np.arccos, sp.acos, 1)
    arctan = register_operator(np.arctan, sp.atan, 1)
    arctan2 = register_operator(np.arctan2, sp.atan2, 2)
    hypot = register_operator(np.hypot, _pyadjoint_hypot, 2)
    sinh = register_operator(np.sinh, sp.sinh, 1)
    cosh = register_operator(np.cosh, sp.cosh, 1)
    tanh = register_operator(np.tanh, sp.tanh, 1)
    arcsinh = register_operator(np.arcsinh, sp.asinh, 1)
    arccosh = register_operator(np.arccosh, sp.acosh, 1)
    arctanh = register_operator(np.arctanh, sp.atanh, 1)
    exp = register_operator(np.exp, sp.exp, 1)
    exp2 = register_operator(np.exp2, lambda x: 2 ** x, 1)
    expm1 = register_operator(np.expm1, _pyadjoint_expm1, 1)
    log = register_operator(np.log, sp.log, 1)
    log2 = register_operator(np.log2, lambda x: sp.log(x, 2), 1)
    log10 = register_operator(np.log10, lambda x: sp.log(x, 10), 1)
    log1p = register_operator(np.log1p, _pyadjoint_log1p, 1)
    sqrt = register_operator(np.sqrt, sp.sqrt, 1)
    square = register_operator(np.square, lambda x: x ** 2, 1)
    cbrt = register_operator(np.cbrt, lambda x: x ** sp.Rational(1, 3), 1)
    reciprocal = register_operator(np.reciprocal, lambda x: sp.Integer(1) / x, 1)

    def _ad_init_zero(self, dual=False):
        return type(self)(0.)

    def _ad_convert_riesz(self, value, riesz_map=None):
        if riesz_map is not None:
            raise ValueError(f"Unexpected Riesz map for Adjfloat: {riesz_map}")
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

    @property
    def _ad_str(self):
        """Return the string of the taped value of this variable."""
        return str(self.block_variable.saved_output)

    def _ad_to_petsc(self, vec=None):
        raise NotImplementedError("_ad_to_petsc not implemented for AdjFloat.")

    def _ad_from_petsc(self, vec):
        raise NotImplementedError("_ad_from_petsc not implemented for AdjFloat.")


# Backwards compatibility
exp = np.exp
log = np.log
min = np.minimum
max = np.maximum
