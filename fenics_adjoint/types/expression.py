import backend
import ufl
from six import add_metaclass

from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.tape import get_working_tape
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
_IGNORED_EXPRESSION_ATTRIBUTES = ['_ufl_is_evaluation_', '__lt__', '__mul__', '_ufl_terminal_modifiers_', '_ufl_num_typecodes_', '__disown__', '_ufl_obj_init_counts_', '_ad_restore_at_checkpoint', 'ufl_shape', 'tape', '__dir__', '__call__', '_ufl_shape', '_ufl_is_differential_', '__init__', '_count', '__doc__', '__sizeof__', 'ufl_free_indices', '__sub__', 'ufl_enable_profiling', '__add__', '__module__', 'cppcode', 'thisown', '__init_subclass__', '__bool__', 'id', '__radd__', '__ne__', '_ufl_function_space', '__swig_destroy__', 'ufl_evaluate', '__rpow__', 'annotate_tape', 'value_shape', '_ufl_evaluate_scalar_', '_ufl_is_shaping_', '__slots__', '__neg__', 'ufl_domain', '_ufl_is_restriction_', '__reduce__', 'name', '_ufl_compute_hash_', '_ufl_obj_del_counts_', '_ufl_required_methods_', '__gt__', '_ufl_typecode_', 'label', '_ufl_regular__del__', 'eval_cell', '_ufl_is_literal_', '_ufl_language_operators_', 'get_adj_output', '__rtruediv__', '__setattr__', '__floordiv__', 'ufl_domains', '__new__', 'value_size', '__getitem__', 'value_dimension', '_ufl_profiling__init__', '__subclasshook__', 'ufl_index_dimensions', '_ufl_class_', '_ufl_is_abstract_', 'create_block_output', '__pow__', 'get_block_output', '__getnewargs__', '__rsub__', '_ufl_err_str_', 'ufl_element', '__repr__', 'geometric_dimension', 'block_output', '__delattr__', '_repr', '_ufl_is_scalar_', 'str', '__abs__', '_ufl_is_index_free_', '_ad_ignored_attributes', 'this', '__float__', '__del__', '_ufl_is_terminal_', '__len__', 'ufl_function_space', '_ad_dot', '__rdiv__', 'is_cellwise_constant', 'value_rank', '_ufl_regular__init__', 'evaluate', '__dict__', '__eq__', '__unicode__', 'reset_variables', 'rename', 'user_defined_derivatives', '_repr_png_', 'count', 'user_parameters', '__truediv__', '__reduce_ex__', '_ad_initialized', '_repr_latex_', 'parameters', '__hash__', 'restrict', 'original_block_output', '_ad_attributes_dict', 'ufl_operands', '__iter__', '_globalcount', '_ufl_expr_reconstruct_', '_ufl_all_classes_', 'eval', 'set_initial_adj_input', '_ufl_required_properties_', '_ufl_signature_data_', 'get_derivative', 'set_block_output', 'update', '__getattribute__', '_ad_create_checkpoint', '_function_space', '__weakref__', '__format__', '_ad_add', '__pos__', '_ufl_profiling__del__', '__rmul__', '_hash', 'adj_update_value', 'compute_vertex_values', 'T', '__class__', '_ufl_is_terminal_modifier_', 'ufl_disable_profiling', '__div__', '__le__', '_ufl_handler_name_', '_value_shape', '__round__', '_ufl_is_in_reference_frame_', 'dx', '_ufl_all_handler_names_', '__str__', '__xor__', '_ufl_coerce_', '_ufl_num_ops_', '__ge__', '__nonzero__', 'set_initial_tlm_input', '_ad_mul', '_ufl_noslots_']

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
            and bases[3] == OverloadedType):

            return type.__new__(mcs, class_name, bases, dict_)

        # Now we need to remove our overloaded Expression from bases,
        # and add the backend Expression type.
        bases = list(bases)
        bases.remove(Expression)
        bases.append(backend.Expression)

        # The if-test might be redundant as users should never define
        # Expression subclasses which inherit from OverloadedType.
        # In fact it might just be better to raise an error if it does.
        if OverloadedType not in bases:
            # Time to add our OverloadedType as a baseclass,
            # as well as its constructor to the
            # new class constructor.
            bases.append(OverloadedType)

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
        # subclass that inherit from OverloadedType.
        original = _backend_ExpressionMetaClass.__new__(mcs, class_name, bases, dict_)

        original_init = original.__dict__["__init__"]

        bases = list(original.__bases__)
        bases[0] = Expression # Replace backend.Expression with our overloaded expression.

        def __init__(self, *args, **kwargs):
            self._ad_initialized = False
            
            Expression.__init__(self, *args, **kwargs)

            OverloadedType.__init__(self, *args, **kwargs)
            original_init(self, *args, **kwargs)
            
            self._ad_initialized = True

            self.annotate_tape = kwargs.pop("annotate_tape", True)
            if self.annotate_tape:
                tape = get_working_tape()
                block = ExpressionBlock(self)
                tape.add_block(block)
                block.add_output(self.block_output)


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
    bases = (Expression, original_bases[1], original_bases[2], OverloadedType)

    original_init = type(original).__dict__["__init__"]

    def __init__(self, cppcode, *args, **kwargs):
        """Init method of our overloaded CompiledExpression classes.

        """
        self._ad_initialized = False
        Expression.__init__(self, *args, **kwargs)
        
        OverloadedType.__init__(self, *args, **kwargs)
        original_init(self, cppcode, *args, **kwargs)

        self._ad_initialized = True

        self.annotate_tape = kwargs.pop("annotate_tape", True)
        if self.annotate_tape:            
            tape = get_working_tape()
            block = ExpressionBlock(self)
            tape.add_block(block)
            block.add_output(self.block_output)

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
            if self._ad_initialized and self.annotate_tape:
                self.block_output.save_output()
                self._ad_attributes_dict[k] = v
                    
                tape = get_working_tape()
                block = ExpressionBlock(self)
                tape.add_block(block)
                block.add_output(self.create_block_output())
            else:
                self._ad_attributes_dict[k] = v
        backend.Expression.__setattr__(self, k, v)

    def _ad_create_checkpoint(self):
        return self._ad_attributes_dict.copy()

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
    def __str__(self):
        return "Expression block"
