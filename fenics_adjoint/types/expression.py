import backend
from pyadjoint.tape import OverloadedType, get_working_tape, Block
import ufl

# TODO: After changes to how Expressions are overloaded, read through the docstrings
#       and fix the inaccuracies. Also add docstrings where they are missing.

# TODO: _ad_initialized might be used more generally to avoid multiple unused ExpressionBlocks
#       when chaning multiple Expression attributes.

# Since we use dir to find every attribute (and method) we get a lot more than just those that will
# probably ever be sent to __setattr__. However this saves us from making this manually. An idea might be to filter out methods,
# but that will be something for a later time.
# Might also consider moving this to a separate file if it ruins this file.
_IGNORED_EXPRESSION_ATTRIBUTES = ['_ufl_is_restriction_', '__format__', '_repr_latex_', '__str__', '_ufl_terminal_modifiers_', 'ufl_function_space', '__rdiv__', '__rmul__', '__lt__', '__weakref__', '_ufl_profiling__del__', 'parameters', 'get_block_output', 'ufl_index_dimensions', '__class__', 'ufl_shape', 'ufl_evaluate', 'tape', '__call__', '_ad_restore_at_checkpoint', 'ufl_enable_profiling', '__getitem__', 'evaluate', 'thisown', 'user_defined_derivatives', 'value_rank', '__subclasshook__', 'T', 'dx', '__gt__', '__bool__', '__round__', '__nonzero__', 'name', 'set_block_output', '_repr', '_ufl_language_operators_', '_ad_attributes_dict', '_ufl_regular__del__', '__module__', '__metaclass__', 'create_block_output', '_ufl_obj_del_counts_', '_ufl_is_differential_', '_ufl_num_typecodes_', '__dict__', '__truediv__', '_ufl_required_properties_', 'annotate_tape', '__getnewargs__', 'ufl_domain', '_ufl_is_shaping_', '__setattr__', 'label', '__ne__', '_ufl_noslots_', '__floordiv__', 'is_cellwise_constant', 'ufl_disable_profiling', '_globalcount', '__unicode__', 'value_dimension', '_ufl_shape', 'ufl_operands', '__xor__', '__delattr__', '_repr_png_', '__repr__', '_hash', '_ufl_function_space', '_ufl_is_scalar_', '__disown__', 'eval', 'ufl_element', '_ufl_num_ops_', '_ufl_class_', '_ufl_required_methods_', '_ufl_signature_data_', 'restrict', '_ufl_coerce_', '__rsub__', '__float__', '__rpow__', '_ufl_err_str_', '_ufl_handler_name_', '__abs__', 'block_output', '_ad_initialized', 'ufl_domains', '_ufl_is_abstract_', '__doc__', '__len__', 'value_size', '_ufl_is_in_reference_frame_', '__del__', '__reduce__', '_value_shape', '__iter__', '__eq__', 'count', '_ufl_is_terminal_', '__swig_destroy__', 'this', 'original_block_output', 'cppcode', '__le__', 'compute_vertex_values', 'str', '_ufl_evaluate_scalar_', '__hash__', '__sub__', 'reset_variables', '__ge__', 'rename', 'ufl_free_indices', '__rtruediv__', '_ufl_compute_hash_', '_ad_create_checkpoint', '__getattribute__', '_ufl_all_handler_names_', '__radd__', 'get_adj_output', '_function_space', '__sizeof__', '_ufl_is_literal_', 'id', '__init__', '__reduce_ex__', '__new__', '_ufl_obj_init_counts_', 'set_initial_adj_input', '_ufl_all_classes_', '__div__', '_ufl_expr_reconstruct_', '__pos__', '_ufl_regular__init__', '__mul__', 'value_shape', '_ad_ignored_attributes', 'update', '_ufl_typecode_', '__add__', '_ufl_profiling__init__', '_count', 'eval_cell', '_ufl_is_terminal_modifier_', '_ufl_is_evaluation_', 'user_parameters', 'geometric_dimension', '__slots__', '__neg__', '_ufl_is_index_free_', '__pow__']

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
                self.block_output.adj_value = []


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
            self.block_output.adj_value = []

    return type.__new__(OverloadedExpressionMetaClass,
                        "CompiledExpression",
                        bases,
                        {"__init__": __init__ })


class Expression(backend.Expression):
    """Overloaded Expression class.

    This class acts as a base class where backend.Expression would be a base class.

    """
    __metaclass__ = OverloadedExpressionMetaClass
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
                self.block_output.adj_value = []
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

        for key in expression._ad_attributes_dict:
            parameter = expression._ad_attributes_dict[key]
            if isinstance(parameter, OverloadedType):
                self.add_dependency(parameter.block_output)

    def evaluate_adj(self):
        adj_inputs = self.get_outputs()[0].get_adj_output()

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
                if isinstance(c, backend.Constant):
                    adj_output = adj_input.inner(interp.vector())
                else:
                    adj_output = adj_input*interp.vector()

                block_output.add_adj_output(adj_output)
