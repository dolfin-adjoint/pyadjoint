import backend
import ufl
from six import add_metaclass

from pyadjoint.overloaded_type import OverloadedType, FloatingType
from pyadjoint.tape import get_working_tape, annotate_tape
from pyadjoint.block import Block
from pyadjoint.adjfloat import AdjFloat
from . import Constant

# TODO: After changes to how Expressions are overloaded, read through the docstrings
#       and fix the inaccuracies. Also add docstrings where they are missing.

# TODO: _ad_initialized might be used more generally to avoid multiple unused ExpressionBlocks
#       when chaning multiple Expression attributes.

# Since we use dir to find every attribute (and method) we get a lot more than just those that will
# probably ever be sent to __setattr__. However this saves us from making this manually. An idea might be to filter out methods,
# but that will be something for a later time.
# Might also consider moving this to a separate file if it ruins this file.
_IGNORED_EXPRESSION_ATTRIBUTES = ['_ad_dim', '__init_subclass__', '_ufl_required_properties_', '__radd__', '__rsub__', 'ufl_evaluate', 'rename', 'geometric_dimension', 'ufl_enable_profiling', '_ufl_profiling__del__', 'adj_update_value', '_ad_outputs', '__getnewargs__', '_ufl_is_index_free_', '_ufl_num_typecodes_', '_ufl_is_restriction_', '_repr_png_', '__getitem__', '_ufl_noslots_', '_ufl_num_ops_', '__subclasshook__', '__pos__', '__sub__', 'ufl_domain', '__getattribute__', '_repr', '__weakref__', '_ufl_is_scalar_', '__delattr__', '__sizeof__', 'evaluate', '_ufl_err_str_', '__module__', 'get_generic_function', '__rdiv__', '_ufl_is_shaping_', '__init__', 'update', 'tlm_value', '_ad_restore_at_checkpoint', '_ad_mul', '__format__', '__swig_destroy__', '_ad_annotate_block', '_ad_will_add_as_output', '_ufl_language_operators_', '_ufl_shape', '_ad_add', 'get_property', '__ne__', 'block_class', 'user_parameters', '__call__', '__reduce_ex__', '__xor__', 'id', '_ad_attributes_dict', 'ufl_shape', '__str__', 'annotate_tape', 'value_dimension', '__bool__', 'output_block_class', '_count', '__round__', '__abs__', '_ad_output_kwargs', 'restrict', '_globalcount', '_ad_to_list', '__iter__', '__disown__', '_ufl_compute_hash_', '_ad_dot', '_ufl_function_space', 'ufl_disable_profiling', '_ad_function_space', 'eval_cell', '_ad_annotate_output_block', 'ufl_free_indices', '_ufl_profiling__init__', '__add__', '_ufl_is_differential_', 'create_block_variable', '_ad_assign_numpy', '__ge__', '_value_shape', '_ad_output_args', '__dict__', '_ufl_handler_name_', '__reduce__', 'dx', 'value_shape', 'eval', 'stop_floating', '_ufl_required_methods_', 'is_cellwise_constant', '__pow__', 'this', '__doc__', 'str', '_ad_initialized', 'user_defined_derivatives', 'thisown', '_ufl_is_abstract_', '__len__', '__slots__', '__class__', '_ufl_is_terminal_', '__rpow__', '__lt__', '__div__', '__del__', '__unicode__', '_ufl_obj_init_counts_', 'value_size', '_ad_kwargs', '__repr__', '_ufl_obj_del_counts_', '_ufl_evaluate_scalar_', '__neg__', 'ufl_domains', 'count', 'cppcode', '__eq__', '_ad_convert_type', 'ad_ignored_attributes', 'set_property', 'label', '_repr_latex_', 'name', '__le__', '__nonzero__', '_ad_copy', '_ad_floating_active', 'output_block', '_ufl_is_evaluation_', 'original_block_variable', '__truediv__', 'adj_value', '__float__', '__hash__', '_ufl_typecode_', '_ufl_all_classes_', '_ufl_is_terminal_modifier_', 'T', 'compute_vertex_values', 'ufl_index_dimensions', '__floordiv__', '__dir__', 'value_rank', 'set_generic_function', 'ufl_function_space', '_ufl_terminal_modifiers_', '__new__', '_ad_create_checkpoint', '_ad_args', '_ufl_signature_data_', '_ad_will_add_as_dependency', 'block', 'block_variable', '_hash', '_ufl_is_literal_', '_function_space', '_ufl_coerce_', 'ufl_element', 'parameters', '__setattr__', '_ufl_class_', '_ufl_is_in_reference_frame_', '_ufl_expr_reconstruct_', '_ufl_all_handler_names_', '_ufl_regular__init__', '__rmul__', '__rtruediv__', '__gt__', '_ufl_regular__del__', 'ufl_operands', '__mul__', '_cpp_object', '_parameters', '_cppcode', '_user_parameters']

_REMEMBERED_EXPRESSION_ATTRIBUTES = ["block_variable", "block", "block_class", "annotate_tape","user_defined_derivatives","ad_ignored_attributes"]

class CompiledExpression(backend.CompiledExpression, FloatingType):

    def __init__(self, *args, **kwargs):
        self._ad_attributes_dict = {}
        self.ad_ignored_attributes = []
        self.user_defined_derivatives = {}
        self.annotate_tape = annotate_tape(kwargs)
        kwarg_copy = {arg:"" for arg in kwargs.keys()}
        for key in kwargs.keys():
            if isinstance(kwargs[key], Constant):
                kwarg_copy[key] = kwargs[key]._cpp_object
            else:
                kwarg_copy[key] = kwargs[key]

        backend.CompiledExpression.__init__(self, *args, **kwarg_copy)
        FloatingType.__init__(self, *args,
                              block_class=ExpressionBlock,
                              annotate=self.annotate_tape,
                            _ad_args=[self],
                              _ad_floating_active=True,
                              **kwargs)
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        
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
            super(backend.CompiledExpression,self).__setattr__(k, checkpoint[k])
        return self 

    def __setattr__(self, k, v):
        # TODO: Maybe only add to dict if annotation is enabled?
        if k not in _IGNORED_EXPRESSION_ATTRIBUTES:
            self._ad_attributes_dict[k] = v
        super(backend.CompiledExpression,self).__setattr__(k, v)

   
class UserExpression(backend.UserExpression, FloatingType):
    """Overloaded UserExpression class.

    This class acts as a base class where backend.UserExpression would be a base class.

    """
    def __init__(self, *args, **kwargs):
        self._ad_attributes_dict = {}
        self.ad_ignored_attributes = []
        self.user_defined_derivatives = {}
        self.annotate_tape = annotate_tape(kwargs)
        backend.UserExpression.__init__(self, *args, **kwargs)
        FloatingType.__init__(self,
                              *args,
                              block_class=ExpressionBlock,
                              annotate=self.annotate_tape,
                              _ad_args=[self],
                              _ad_floating_active=True,
                              **kwargs)

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
            backend.UserExpression.__setattr__(self, k, checkpoint[k])
        return self 

    def __setattr__(self, k, v):
        # TODO: Maybe only add to dict if annotation is enabled?
        if k not in _IGNORED_EXPRESSION_ATTRIBUTES:
            self._ad_attributes_dict[k] = v
        backend.UserExpression.__setattr__(self, k, v)

class Expression(backend.Expression, FloatingType):
    """Overloaded Expression class.

    This class acts as a base class where backend.Expression would be a base class.

    """
    def __init__(self, *args, **kwargs):
        self._ad_attributes_dict = {}
        self.ad_ignored_attributes = []
        self.user_defined_derivatives = {}
        self.annotate_tape = annotate_tape(kwargs)
        backend.Expression.__init__(self, *args, **kwargs)
        FloatingType.__init__(self,
                              *args,
                              block_class=ExpressionBlock,
                              annotate=self.annotate_tape,
                              _ad_args=[self],
                              _ad_floating_active=True,
                              **kwargs)
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

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

    def __setattr__(self, k, v):
        # TODO: Maybe only add to dict if annotation is enabled?
        if k not in _IGNORED_EXPRESSION_ATTRIBUTES:
            self._ad_attributes_dict[k] = v

        if k in _REMEMBERED_EXPRESSION_ATTRIBUTES:
            object.__setattr__(self, k, v)
        else:
            backend.Expression.__setattr__(self, k, v)



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

    def evaluate_adj(self):
        adj_inputs = self.get_outputs()[0].adj_value

        if adj_inputs is None:
            # No adjoint inputs, so nothing to compute.
            return

        for block_variable in self.get_dependencies():
            c = block_variable.output

            if c not in self.expression.user_defined_derivatives:
                continue

            for adj_pair in adj_inputs:
                adj_input = adj_pair[0]
                V = adj_pair[1]

                for key in self.expression._ad_attributes_dict:
                    if key not in self.expression.ad_ignored_attributes:
                        setattr(self.expression.user_defined_derivatives[c], key, self.expression._ad_attributes_dict[key])

                interp = backend.interpolate(self.expression.user_defined_derivatives[c], V)
                if isinstance(c, (backend.Constant, AdjFloat)):
                    adj_output = adj_input.inner(interp.vector())
                    block_variable.add_adj_output(adj_output)
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
                    block_variable.add_adj_output(adj_output)

    def evaluate_tlm(self):
        output = self.get_outputs()[0]
        # Restore _ad_attributes_dict.
        output.saved_output

        for block_variable in self.get_dependencies():
            if block_variable.tlm_value is None:
                continue

            c = block_variable.output
            if c not in self.expression.user_defined_derivatives:
                continue

            for key in self.expression._ad_attributes_dict:
                if key not in self.expression.ad_ignored_attributes:
                    setattr(self.expression.user_defined_derivatives[c], key, self.expression._ad_attributes_dict[key])

            tlm_input = block_variable.tlm_value

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
                if key not in self.expression.ad_ignored_attributes:
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
                    if key not in self.expression.ad_ignored_attributes:
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
            for block_variable in self.get_dependencies():
                key = self.dependency_keys[block_variable.output]
                checkpoint[key] = block_variable.saved_output

    def __str__(self):
        return "Expression block"
