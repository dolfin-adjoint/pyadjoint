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
_IGNORED_EXPRESSION_ATTRIBUTES = ['_ad_convert_type', '_ad_function_space', '__gt__', '__ne__', '__div__', '__rdiv__', 'is_cellwise_constant', 'parameters', '_function_space', 'user_parameters', 'adj_update_value', '_ad_annotate_output_block', 'value_shape', 'reset_variables', '_ad_will_add_as_output', '_ufl_evaluate_scalar_', 'ufl_free_indices', '__del__', '_ad_floating_active', '__ge__', '__nonzero__', '__rsub__', 'dx', '_ad_restore_at_checkpoint', '_ad_kwargs', 'this', '__doc__', '__repr__', '__bool__', '__setattr__', 'stop_floating', 'ufl_evaluate', '__rtruediv__', '_ufl_is_differential_', '__call__', '__abs__', 'ufl_disable_profiling', 'geometric_dimension', '__unicode__', 'get_block_output', 'tape', '__rpow__', '_ufl_is_in_reference_frame_', '_ufl_num_typecodes_', 'compute_vertex_values', 'ufl_element', '__init_subclass__', 'name', 'ufl_shape', 'block_output', '__weakref__', '__pow__', '_ad_attributes_dict', 'ufl_operands', '__radd__', 'thisown', '_ufl_is_terminal_', '_ufl_err_str_', 'set_block_output', 'ufl_domains', '_ufl_all_classes_', '__rmul__', '__mul__', '__add__', '_ufl_is_shaping_', '__delattr__', '_ufl_obj_del_counts_', 'T', '_count', '_ufl_profiling__del__', '_ad_outputs', 'value_dimension', 'annotate_tape', 'label', '_ad_will_add_as_dependency', 'get_adj_output', '__lt__', 'rename', 'cppcode', '__truediv__', '_ufl_all_handler_names_', '_repr', '_ufl_is_evaluation_', '_ad_mul', 'block_class', 'output_block', '__new__', '__hash__', '__eq__', '_ad_annotate_block', '_hash', '__sub__', '_ufl_required_methods_', '_value_shape', '_ufl_is_scalar_', 'id', '__disown__', '__subclasshook__', '__getitem__', '_ufl_language_operators_', '__iter__', 'set_initial_tlm_input', '_ad_dot', '_ufl_is_index_free_', '_ufl_is_restriction_', '_ufl_regular__init__', 'ufl_function_space', '__init__', '_ad_output_kwargs', '__le__', '__floordiv__', '__round__', 'ufl_index_dimensions', '__str__', '__module__', '_ufl_typecode_', 'ufl_enable_profiling', '_ad_add', '_ufl_compute_hash_', '_ufl_is_abstract_', '_ad_ignored_attributes', '__class__', '_repr_latex_', '__dir__', 'count', '__slots__', 'block', '__swig_destroy__', '_ufl_is_literal_', 'eval_cell', '_globalcount', '__neg__', '__len__', '_ufl_required_properties_', '__getnewargs__', '__format__', '_ufl_class_', '__pos__', 'evaluate', '_ufl_noslots_', 'ufl_domain', '_ad_output_args', 'value_rank', '_ad_initialized', '_ad_create_checkpoint', '__xor__', '_ufl_coerce_', 'value_size', '_ufl_expr_reconstruct_', '_ufl_terminal_modifiers_', 'user_defined_derivatives', '__dict__', '_repr_png_', '_ufl_obj_init_counts_', 'output_block_class', 'update', '_ufl_shape', '_ufl_function_space', '__reduce_ex__', 'str', '__float__', '_ad_args', '_ufl_num_ops_', '_ufl_is_terminal_modifier_', '__sizeof__', 'restrict', '__getattribute__', '_ufl_signature_data_', '__reduce__', 'original_block_output', 'set_initial_adj_input', '_ufl_handler_name_', 'create_block_output', '_ufl_regular__del__', '_ufl_profiling__init__', 'eval']

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
        self._ad_ignored_attributes = []
        self.user_defined_derivatives = {}

    def __setattr__(self, k, v):
        # TODO: Maybe only add to dict if annotation is enabled?
        if k not in _IGNORED_EXPRESSION_ATTRIBUTES:
            self._ad_attributes_dict[k] = v
        backend.Expression.__setattr__(self, k, v)

    def _ad_function_space(self, mesh):
        element = self.ufl_element()
        args = [element.family(), mesh.ufl_cell(), element.degree()]

        # TODO: In newer versions of FEniCS we may use FiniteElement.reconstruct and avoid the if-test
        #       and just write element.reconstruct(cell=mesh.ufl_cell()).
        if isinstance(element, backend.VectorElement):
            fs_element = element.__class__(*args, dim=len(element.sub_elements()))
        else:
            fs_element = element.__class__(*args)
        return backend.FunctionSpace(mesh, fs_element)

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

            if c not in self.expression.user_defined_derivatives:
                continue

            for adj_pair in adj_inputs:
                adj_input = adj_pair[0]
                V = adj_pair[1]

                for key in self.expression._ad_attributes_dict:
                    # TODO: If _ad_ignored_attributes is used directly by the user. Should it not have the _ad_ prefix?
                    if key not in self.expression._ad_ignored_attributes:
                        setattr(self.expression.user_defined_derivatives[c], key, self.expression._ad_attributes_dict[key])

                interp = backend.interpolate(self.expression.user_defined_derivatives[c], V)
                if isinstance(c, (backend.Constant, AdjFloat)):
                    adj_output = adj_input.inner(interp.vector())
                    block_output.add_adj_output(adj_output)
                else:
                    adj_output = adj_input*interp.vector()

                    adj_func = backend.Function(V, adj_output)
                    adj_output = 0
                    num_sub_spaces = V.num_sub_spaces()
                    if num_sub_spaces > 1:
                        for i in range(num_sub_spaces):
                            adj_output += backend.interpolate(adj_func.sub(i), c.function_space()).vector()
                    else:
                        adj_output = backend.interpolate(adj_func, c.function_space()).vector()
                    block_output.add_adj_output(adj_output)

    def evaluate_tlm(self):
        output = self.get_outputs()[0]
        # Restore _ad_attributes_dict.
        output.get_saved_output()

        for block_output in self.get_dependencies():
            if block_output.tlm_value is None:
                continue

            c = block_output.output
            if c not in self.expression.user_defined_derivatives:
                continue

            for key in self.expression._ad_attributes_dict:
                if key not in self.expression._ad_ignored_attributes:
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
                if key not in self.expression._ad_ignored_attributes:
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
                    if key not in self.expression._ad_ignored_attributes:
                        setattr(second_deriv, key, self.expression._ad_attributes_dict[key])

                for adj_pair in adj_inputs:
                    adj_input = adj_pair[0]
                    V = adj_pair[1]

                    # TODO: Seems we can only project and not interpolate ufl.algebra.Product in dolfin.
                    #       Consider the difference and which actually makes sense here.
                    interp = backend.project(tlm_input*second_deriv, V)
                    if isinstance(c1, (backend.Constant, AdjFloat)):
                        hessian_output = adj_input.inner(interp.vector())
                        bo1.add_hessian_output(hessian_output)
                    else:
                        hessian_output = adj_input * interp.vector()

                        hessian_func = backend.Function(V, hessian_output)
                        hessian_output = 0
                        num_sub_spaces = V.num_sub_spaces()
                        if num_sub_spaces > 1:
                            for i in range(num_sub_spaces):
                                hessian_output += backend.interpolate(hessian_func.sub(i), c1.function_space()).vector()
                        else:
                            hessian_output = backend.interpolate(hessian_func, c1.function_space()).vector()
                        bo1.add_hessian_output(hessian_output)

            for hessian_pair in hessian_inputs:
                hessian_input = hessian_pair[0]
                V = hessian_pair[1]

                interp = backend.interpolate(first_deriv, V)
                if isinstance(c1, (backend.Constant, AdjFloat)):
                    hessian_output = hessian_input.inner(interp.vector())
                    bo1.add_hessian_output(hessian_output)
                else:
                    hessian_output = hessian_input*interp.vector()

                    hessian_func = backend.Function(V, hessian_output)
                    hessian_output = 0
                    num_sub_spaces = V.num_sub_spaces()
                    if num_sub_spaces > 1:
                        for i in range(num_sub_spaces):
                            hessian_output += backend.interpolate(hessian_func.sub(i), c1.function_space()).vector()
                    else:
                        hessian_output = backend.interpolate(hessian_func, c1.function_space()).vector()
                    bo1.add_hessian_output(hessian_output)

    def recompute(self):
        checkpoint = self.get_outputs()[0].checkpoint

        if checkpoint:
            for block_output in self.get_dependencies():
                key = self.dependency_keys[block_output.output]
                checkpoint[key] = block_output.get_saved_output()

    def __str__(self):
        return "Expression block"
