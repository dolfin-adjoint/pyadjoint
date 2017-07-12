import backend
import ufl
from six import add_metaclass

from pyadjoint.overloaded_type import OverloadedType, FloatingType
from pyadjoint.tape import get_working_tape, annotate_tape
from pyadjoint.block import Block
from pyadjoint.adjfloat import AdjFloat

# TODO: After changes to how Expressions are overloaded, read through the docstrings
#       and fix the inaccuracies. Also add docstrings where they are missing.

# TODO: _ad_initialized might be used more generally to avoid multiple unused ExpressionBlocks
#       when chaning multiple Expression attributes.

# Since we use dir to find every attribute (and method) we get a lot more than just those that will
# probably ever be sent to __setattr__. However this saves us from making this manually. An idea might be to filter out methods,
# but that will be something for a later time.
# Might also consider moving this to a separate file if it ruins this file.
_IGNORED_EXPRESSION_ATTRIBUTES = ['__hash__', '__class__', 'set_initial_tlm_input', '__disown__', '__le__', '_ufl_profiling__del__', 'get_derivative', 'tape', '__rdiv__', '__eq__', '_ad_restore_at_checkpoint', 'original_block_output', '_ad_ignored_attributes', '_ufl_obj_init_counts_', '_hash', 'ufl_disable_profiling', '__reduce__', '__bool__', 'id', '__slots__', '_ufl_class_', '__rtruediv__', 'adj_update_value', '_globalcount', 'compute_vertex_values', '__swig_destroy__', 'value_dimension', '_ad_create_checkpoint', '__getattribute__', '_ufl_language_operators_', 'this', '__rmul__', '_ad_add', '_ad_attributes_dict', '__getnewargs__', '_ad_dot', '_ufl_is_scalar_', '_ufl_terminal_modifiers_', '_ad_mul', '_ufl_obj_del_counts_', '_ufl_required_methods_', 'parameters', '_ufl_coerce_', '__neg__', '__weakref__', '_ad_initialized', '_ufl_is_shaping_', '__sub__', 'update', 'ufl_element', 'value_size', '_repr_png_', 'annotate_tape', '_ufl_is_terminal_modifier_', '__round__', '__call__', 'block', '_ufl_evaluate_scalar_', '__div__', '__reduce_ex__', '__mul__', '__xor__', '_ufl_handler_name_', '_ufl_regular__init__', '__add__', '_ufl_required_properties_', '__dict__', '_ad_annotate_block', '__dir__', '__abs__', '_ufl_function_space', '__unicode__', 'user_parameters', '_ufl_all_handler_names_', '__radd__', 'ufl_evaluate', '_ufl_regular__del__', '__module__', '_ufl_shape', 'eval_cell', 'thisown', '_ufl_is_terminal_', 'block_class', 'reset_variables', 'evaluate', 'ufl_shape', '__iter__', 'restrict', '_ad_floating_active', '_ad_args', '_ufl_signature_data_', '__doc__', '__delattr__', 'get_adj_output', '__rsub__', 'str', '_ufl_is_literal_', '__pos__', 'create_block_output', '_repr_latex_', '_ufl_is_evaluation_', '_ufl_is_in_reference_frame_', '__pow__', 'ufl_function_space', 'set_block_output', 'dx', 'name', '__getitem__', 'user_defined_derivatives', '_ad_kwargs', 'rename', 'T', 'ufl_operands', '__repr__', '_ufl_is_restriction_', 'label', '_count', '_ufl_all_classes_', '_ufl_profiling__init__', '__lt__', '_ufl_is_index_free_', '__floordiv__', 'ufl_domain', '__ne__', '_ufl_compute_hash_', '_ufl_num_ops_', 'block_output', '__truediv__', '__sizeof__', '__del__', 'value_shape', '__new__', '_ufl_typecode_', 'ufl_free_indices', 'eval', 'ufl_domains', '_function_space', '_ufl_err_str_', 'ufl_index_dimensions', '_value_shape', '__ge__', '__init_subclass__', '__format__', 'cppcode', '_ufl_noslots_', 'count', '__str__', 'value_rank', '__setattr__', '__subclasshook__', 'is_cellwise_constant', '_ufl_num_typecodes_', 'geometric_dimension', '_repr', '__float__', '_ufl_is_differential_', 'get_block_output', '__len__', '__init__', '__nonzero__', '_ufl_is_abstract_', '__gt__', 'ufl_enable_profiling', 'set_initial_adj_input', '__rpow__', '_ufl_expr_reconstruct_']

_backend_ExpressionMetaClass = backend.functions.expression.ExpressionMetaClass

class OverloadedExpressionMetaClass(_backend_ExpressionMetaClass):
    """Overloads the ExpressionMetaClass so that we can create overloaded 
    Expression types.

    """
    def __new__(mcs, class_name, bases, dict_):
        """Handles creation of new Expression classes.

        """
        if class_name == "Expression" or class_name == "CompiledExpression":
            if len(bases) >= 1 and bases[0] == backend.Expression:
                # If this is our own overloaded Expression/CompiledExpression, then we only want to use
                # our own definition, and not call ExpressionMetaClass.__new__.
                return type.__new__(mcs, class_name, bases, dict_)


        if (len(bases) >= 4
            and bases[0] == Expression
            and bases[1] == ufl.Coefficient
            and issubclass(bases[2], backend.cpp.Expression)
            and bases[3] == FloatingType):

            return type.__new__(mcs, class_name, bases, dict_)

        # Now we need to remove our overloaded Expression from bases,
        # and add the backend Expression type.
        bases = list(bases)
        bases.remove(Expression)
        bases.append(backend.Expression)

        # The if-test might be redundant as users should never define
        # Expression subclasses which inherit from FloatingType.
        # In fact it might just be better to raise an error if it does.
        if FloatingType not in bases:
            # Time to add our FloatingType as a baseclass,
            # as well as its constructor to the
            # new class constructor.
            bases.append(FloatingType)

            user_init = dict_.get("__init__", None)

            if user_init is None:
                def __init__(self, *args, **kwargs):
                    """Overloaded init method of user-defined Expression classes.

                    Workaround for the kwargs check done with no user_init.

                    """
                    pass
                dict_["__init__"] = __init__

        bases = tuple(bases)

        # Pass everything along to the backend metaclass.
        # This should return a new user-defined Expression 
        # subclass that inherit from FloatingType.
        original = _backend_ExpressionMetaClass.__new__(mcs, class_name, bases, dict_)

        original_init = original.__dict__["__init__"]

        bases = list(original.__bases__)
        bases[0] = Expression  # Replace backend.Expression with our overloaded expression.

        def __init__(self, *args, **kwargs):
            self._ad_initialized = False

            Expression.__init__(self, *args, **kwargs)

            FloatingType.__init__(self,
                                  *args,
                                  block_class=ExpressionBlock,
                                  annotate=annotate_tape(kwargs),
                                  _ad_args=[self],
                                  _ad_floating_active=True,
                                  **kwargs)
            original_init(self, *args, **kwargs)

            self._ad_initialized = True

        dict_["__init__"] = __init__
        bases = tuple(bases)

        overloaded = type(class_name, bases, dict_)

        return overloaded

    def __call__(cls, *args, **kwargs):
        """Handles instantiation and initialization of Expression instances.

        """
        # Calls type.__call__ as normal (backend does not override __call__).
        # Thus __new__ of cls is invoked.
        out = _backend_ExpressionMetaClass.__call__(cls, *args, **kwargs)

        # Since the class we create is not a subclass of our overloaded Expression 
        # (where __new__ was invoked), type.__call__ will not initialize
        # the newly created instance. Hence we must do so "manually".
        if not isinstance(out, cls):
            out.__init__(*args, **kwargs)
        return out


def create_compiled_expression(original, cppcode, *args, **kwargs):
    """Creates an overloaded CompiledExpression type from the supplied 
    CompiledExpression instance.

    The argument `original` will be an (uninitialized) CompiledExpression instance,
    which we extract the type from to build our own corresponding overloaded
    CompiledExpression type.

    Args:
        original (:obj:`CompiledExpression`): The original CompiledExpression instance.
        cppcode (:obj:`str`): The cppcode used to define the Expression.
        *args: Extra arguments are just passed along to the backend handlers.
        **kwargs: Keyword arguments are also just passed to backend handlers.

    Returns:
        :obj:`type`: An overloaded CompiledExpression type.

    """
    original_bases = type(original).__bases__
    bases = (Expression, original_bases[1], original_bases[2], FloatingType)

    original_init = type(original).__dict__["__init__"]

    def __init__(self, cppcode, *args, **kwargs):
        """Init method of our overloaded CompiledExpression classes.

        """
        self._ad_initialized = False
        Expression.__init__(self, *args, **kwargs)

        FloatingType.__init__(self,
                              *args,
                              block_class=ExpressionBlock,
                              annotate=annotate_tape(kwargs),
                              _ad_args=[self],
                              _ad_floating_active=True,
                              **kwargs)
        original_init(self, cppcode, *args, **kwargs)

        self._ad_initialized = True

    return type.__new__(OverloadedExpressionMetaClass,
                        "CompiledExpression",
                        bases,
                        {"__init__": __init__ })


@add_metaclass(OverloadedExpressionMetaClass)
class Expression(backend.Expression):
    """Overloaded Expression class.

    This class acts as a base class where backend.Expression would be a base class.

    """
    def __new__(cls, cppcode=None, *args, **kwargs):
        if cls.__name__ != "Expression":
            return object.__new__(cls)

        original = backend.Expression.__new__(cls, cppcode, *args, **kwargs)
        return object.__new__(create_compiled_expression(original, cppcode, *args, **kwargs))

    def __init__(self, *args, **kwargs):
        self._ad_attributes_dict = {}
        self._ad_ignored_attributes = None
        self.user_defined_derivatives = {}

    def __setattr__(self, k, v):    
        if k not in _IGNORED_EXPRESSION_ATTRIBUTES:
            self._ad_attributes_dict[k] = v
        backend.Expression.__setattr__(self, k, v)

    def _ad_create_checkpoint(self):
        ret = {}
        for k in self._ad_attributes_dict:
            v = self._ad_attributes_dict[k]
            if isinstance(v, OverloadedType):
                ret[k] = v.block_output.get_saved_output()
            else:
                ret[k] = v
        return ret

    def _ad_restore_at_checkpoint(self, checkpoint):
        for k in checkpoint:
            self._ad_attributes_dict[k] = checkpoint[k]
            backend.Expression.__setattr__(self, k, checkpoint[k])
        return self


class ExpressionBlock(Block):

    def __init__(self, expression):
        super(ExpressionBlock, self).__init__()
        self.expression = expression
        self.dependency_keys = {}

        for key in expression._ad_attributes_dict:
            parameter = expression._ad_attributes_dict[key]
            if isinstance(parameter, OverloadedType):
                self.add_dependency(parameter.block_output)
                self.dependency_keys[parameter] = key

    def evaluate_adj(self):
        adj_inputs = self.get_outputs()[0].get_adj_output()

        if adj_inputs is None:
            # No adjoint inputs, so nothing to compute.
            return

        for block_output in self.get_dependencies():
            c = block_output.output
            for adj_pair in adj_inputs:
                adj_input = adj_pair[0]
                V = adj_pair[1]

                for key in self.expression._ad_attributes_dict:
                    # TODO: If _ad_ignored_attributes is used directly by the user. Should it not have the _ad_ prefix?
                    if (self.expression._ad_ignored_attributes is None
                        or key not in self.expression._ad_ignored_attributes):

                        setattr(self.expression.user_defined_derivatives[c], key, self.expression._ad_attributes_dict[key])

                interp = backend.interpolate(self.expression.user_defined_derivatives[c], V)
                if isinstance(c, (backend.Constant, AdjFloat)):
                    adj_output = adj_input.inner(interp.vector())
                else:
                    adj_output = adj_input*interp.vector()

                block_output.add_adj_output(adj_output)

    def evaluate_tlm(self):
        output = self.get_outputs()[0]
        # Restore _ad_attributes_dict.
        output.get_saved_output()

        for block_output in self.get_dependencies():
            if block_output.tlm_value is None:
                continue

            c = block_output.output
            for key in self.expression._ad_attributes_dict:
                if (self.expression._ad_ignored_attributes is None
                    or key not in self.expression._ad_ignored_attributes):
                    setattr(self.expression.user_defined_derivatives[c], key, self.expression._ad_attributes_dict[key])

            tlm_input = block_output.tlm_value

            output.add_tlm_output(tlm_input * self.expression.user_defined_derivatives[c])

    def evaluate_hessian(self):
        hessian_inputs = self.get_outputs()[0].hessian_value
        adj_inputs = self.get_outputs()[0].adj_value

        if hessian_inputs is None:
            return

        for bo1 in self.get_dependencies():
            c1 = bo1.output

            if c1 not in self.expression.user_defined_derivatives:
                continue

            first_deriv = self.expression.user_defined_derivatives[c1]
            for key in self.expression._ad_attributes_dict:
                if (self.expression._ad_ignored_attributes is None
                    or key not in self.expression._ad_ignored_attributes):
                    setattr(first_deriv, key, self.expression._ad_attributes_dict[key])

            for bo2 in self.get_dependencies():
                c2 = bo2.output
                tlm_input = bo2.tlm_value

                if tlm_input is None:
                    continue

                if c2 not in first_deriv.user_defined_derivatives:
                    continue

                second_deriv = first_deriv.user_defined_derivatives[c2]
                for key in self.expression._ad_attributes_dict:
                    if (self.expression._ad_ignored_attributes is None
                        or key not in self.expression._ad_ignored_attributes):
                        setattr(second_deriv, key, self.expression._ad_attributes_dict[key])

                for adj_pair in adj_inputs:
                    adj_input = adj_pair[0]
                    V = adj_pair[1]

                    # TODO: Seems we can only project and not interpolate ufl.algebra.Product in dolfin.
                    #       Consider the difference and which actually makes sense here.
                    interp = backend.project(tlm_input*second_deriv, V)
                    if isinstance(c1, (backend.Constant, AdjFloat)):
                        hessian_output = adj_input.inner(interp.vector())
                    else:
                        hessian_output = adj_input * interp.vector()

                    bo1.add_hessian_output(hessian_output)

            for hessian_pair in hessian_inputs:
                hessian_input = hessian_pair[0]
                V = hessian_pair[1]

                interp = backend.interpolate(first_deriv, V)
                if isinstance(c1, (backend.Constant, AdjFloat)):
                    hessian_output = hessian_input.inner(interp.vector())
                else:
                    hessian_output = hessian_input*interp.vector()

                bo1.add_hessian_output(hessian_output)

    def recompute(self):
        checkpoint = self.get_outputs()[0].checkpoint

        if checkpoint:
            for block_output in self.get_dependencies():
                key = self.dependency_keys[block_output.output]
                checkpoint[key] = block_output.get_saved_output()
