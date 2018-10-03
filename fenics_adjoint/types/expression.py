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
_IGNORED_EXPRESSION_ATTRIBUTES = ['_ad_init_object', '_ad_dim', '__init_subclass__', '_ufl_required_properties_', '__radd__', '__rsub__', 'ufl_evaluate', 'rename', 'geometric_dimension', 'ufl_enable_profiling', '_ufl_profiling__del__', 'adj_update_value', '_ad_outputs', '__getnewargs__', '_ufl_is_index_free_', '_ufl_num_typecodes_', '_ufl_is_restriction_', '_repr_png_', '__getitem__', '_ufl_noslots_', '_ufl_num_ops_', '__subclasshook__', '__pos__', '__sub__', 'ufl_domain', '__getattribute__', '_repr', '__weakref__', '_ufl_is_scalar_', '__delattr__', '__sizeof__', 'evaluate', '_ufl_err_str_', '__module__', 'get_generic_function', '__rdiv__', '_ufl_is_shaping_', '__init__', 'update', 'tlm_value', '_ad_restore_at_checkpoint', '_ad_mul', '__format__', '__swig_destroy__', '_ad_annotate_block', '_ad_will_add_as_output', '_ufl_language_operators_', '_ufl_shape', '_ad_add', 'get_property', '__ne__', 'block_class', 'user_parameters', '__call__', '__reduce_ex__', '__xor__', 'id', '_ad_attributes_dict', 'ufl_shape', '__str__', 'annotate_tape', 'value_dimension', '__bool__', 'output_block_class', '_count', '__round__', '__abs__', '_ad_output_kwargs', 'restrict', '_globalcount', '_ad_to_list', '__iter__', '__disown__', '_ufl_compute_hash_', '_ad_dot', '_ufl_function_space', 'ufl_disable_profiling', '_ad_function_space', 'eval_cell', '_ad_annotate_output_block', 'ufl_free_indices', '_ufl_profiling__init__', '__add__', '_ufl_is_differential_', 'create_block_variable', '_ad_assign_numpy', '__ge__', '_value_shape', '_ad_output_args', '__dict__', '_ufl_handler_name_', '__reduce__', 'dx', 'value_shape', 'eval', 'stop_floating', '_ufl_required_methods_', 'is_cellwise_constant', '__pow__', 'this', '__doc__', 'str', '_ad_initialized', 'user_defined_derivatives', 'thisown', '_ufl_is_abstract_', '__len__', '__slots__', '__class__', '_ufl_is_terminal_', '__rpow__', '__lt__', '__div__', '__del__', '__unicode__', '_ufl_obj_init_counts_', 'value_size', '_ad_kwargs', '__repr__', '_ufl_obj_del_counts_', '_ufl_evaluate_scalar_', '__neg__', 'ufl_domains', 'count', 'cppcode', '__eq__', '_ad_convert_type', 'ad_ignored_attributes', 'set_property', 'label', '_repr_latex_', 'name', '__le__', '__nonzero__', '_ad_copy', '_ad_floating_active', 'output_block', '_ufl_is_evaluation_', 'original_block_variable', '__truediv__', 'adj_value', '__float__', '__hash__', '_ufl_typecode_', '_ufl_all_classes_', '_ufl_is_terminal_modifier_', 'T', 'compute_vertex_values', 'ufl_index_dimensions', '__floordiv__', '__dir__', 'value_rank', 'set_generic_function', 'ufl_function_space', '_ufl_terminal_modifiers_', '__new__', '_ad_create_checkpoint', '_ad_args', '_ufl_signature_data_', '_ad_will_add_as_dependency', 'block', 'block_variable', '_hash', '_ufl_is_literal_', '_function_space', '_ufl_coerce_', 'ufl_element', 'parameters', '__setattr__', '_ufl_class_', '_ufl_is_in_reference_frame_', '_ufl_expr_reconstruct_', '_ufl_all_handler_names_', '_ufl_regular__init__', '__rmul__', '__rtruediv__', '__gt__', '_ufl_regular__del__', 'ufl_operands', '__mul__']

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
            # pop annotate from kwargs before passing them on the backend init
            self.annotate_tape = annotate_tape(kwargs)

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
        # pop annotate from kwargs before passing them on the backend init
        self.annotate_tape = annotate_tape(kwargs)

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
        self.ad_ignored_attributes = []
        self.user_defined_derivatives = {}

    def __setattr__(self, k, v):
        # TODO: Maybe only add to dict if annotation is enabled?
        if k not in _IGNORED_EXPRESSION_ATTRIBUTES:
            self._ad_attributes_dict[k] = v
        backend.Expression.__setattr__(self, k, v)

    def _ad_function_space(self, mesh):
        element = self.ufl_element()
        fs_element = element.reconstruct(cell=mesh.ufl_cell())
        return backend.FunctionSpace(mesh, fs_element)

    def _ad_create_checkpoint(self):
        ret = {}
        for k in self._ad_attributes_dict:
            v = self._ad_attributes_dict[k]
            if isinstance(v, OverloadedType):
                ret[k] = v.block_variable.saved_output
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
                self.add_dependency(parameter.block_variable)
                self.dependency_keys[parameter] = key

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_inputs = adj_inputs[0]
        c = block_variable.output
        if c not in self.expression.user_defined_derivatives:
            return None

        for key in self.expression._ad_attributes_dict:
            if key not in self.expression.ad_ignored_attributes:
                setattr(self.expression.user_defined_derivatives[c], key,
                        self.expression._ad_attributes_dict[key])

        adj_output = None
        for adj_pair in adj_inputs:
            adj_input = adj_pair[0]
            V = adj_pair[1]
            if adj_output is None:
                adj_output = 0.0

            interp = backend.interpolate(self.expression.user_defined_derivatives[c], V)
            if isinstance(c, (backend.Constant, AdjFloat)):
                adj_output += adj_input.inner(interp.vector())
            else:
                vec = adj_input * interp.vector()
                adj_func = backend.Function(V, vec)

                num_sub_spaces = V.num_sub_spaces()
                if num_sub_spaces > 1:
                    for i in range(num_sub_spaces):
                        adj_output += backend.interpolate(adj_func.sub(i), c.function_space()).vector()
                else:
                    adj_output += backend.interpolate(adj_func, c.function_space()).vector()
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        # Restore _ad_attributes_dict.
        block_variable.saved_output

        tlm_output = 0.
        tlm_used = False
        for block_variable in self.get_dependencies():
            tlm_input = block_variable.tlm_value
            if tlm_input is None:
                continue

            c = block_variable.output
            if c not in self.expression.user_defined_derivatives:
                continue

            for key in self.expression._ad_attributes_dict:
                if key not in self.expression.ad_ignored_attributes:
                    setattr(self.expression.user_defined_derivatives[c], key, self.expression._ad_attributes_dict[key])

            tlm_used = True
            tlm_output += tlm_input * self.expression.user_defined_derivatives[c]
        if not tlm_used:
            return None
        return tlm_output

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        hessian_inputs = hessian_inputs[0]
        adj_inputs = adj_inputs[0]
        c1 = block_variable.output

        if c1 not in self.expression.user_defined_derivatives:
            return None

        first_deriv = self.expression.user_defined_derivatives[c1]
        for key in self.expression._ad_attributes_dict:
            if key not in self.expression.ad_ignored_attributes:
                setattr(first_deriv, key, self.expression._ad_attributes_dict[key])

        hessian_output = None
        for _, bo2 in relevant_dependencies:
            c2 = bo2.output
            tlm_input = bo2.tlm_value

            if tlm_input is None:
                continue

            if c2 not in first_deriv.user_defined_derivatives:
                continue

            second_deriv = first_deriv.user_defined_derivatives[c2]
            for key in self.expression._ad_attributes_dict:
                if key not in self.expression.ad_ignored_attributes:
                    setattr(second_deriv, key, self.expression._ad_attributes_dict[key])

            for adj_pair in adj_inputs:
                adj_input = adj_pair[0]
                V = adj_pair[1]

                if hessian_output is None:
                    hessian_output = 0.0

                # TODO: Seems we can only project and not interpolate ufl.algebra.Product in dolfin.
                #       Consider the difference and which actually makes sense here.
                interp = backend.project(tlm_input * second_deriv, V)
                if isinstance(c1, (backend.Constant, AdjFloat)):
                    hessian_output += adj_input.inner(interp.vector())
                else:
                    vec = adj_input * interp.vector()
                    hessian_func = backend.Function(V, vec)

                    num_sub_spaces = V.num_sub_spaces()
                    if num_sub_spaces > 1:
                        for i in range(num_sub_spaces):
                            hessian_output += backend.interpolate(hessian_func.sub(i), c1.function_space()).vector()
                    else:
                        hessian_output += backend.interpolate(hessian_func, c1.function_space()).vector()

        for hessian_pair in hessian_inputs:
            if hessian_output is None:
                hessian_output = 0.0
            hessian_input = hessian_pair[0]
            V = hessian_pair[1]

            interp = backend.interpolate(first_deriv, V)
            if isinstance(c1, (backend.Constant, AdjFloat)):
                hessian_output += hessian_input.inner(interp.vector())
            else:
                vec = hessian_input * interp.vector()
                hessian_func = backend.Function(V, vec)

                num_sub_spaces = V.num_sub_spaces()
                if num_sub_spaces > 1:
                    for i in range(num_sub_spaces):
                        hessian_output += backend.interpolate(hessian_func.sub(i), c1.function_space()).vector()
                else:
                    hessian_output += backend.interpolate(hessian_func, c1.function_space()).vector()
        return hessian_output

    def recompute_component(self, inputs, block_variable, idx, prepared):
        checkpoint = self.get_outputs()[0].checkpoint

        if checkpoint:
            for bv in self.get_dependencies():
                key = self.dependency_keys[bv.output]
                checkpoint[key] = bv.saved_output

    def __str__(self):
        return "Expression block"
