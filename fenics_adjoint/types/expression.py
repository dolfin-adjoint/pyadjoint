import backend
from pyadjoint.tape import OverloadedType, get_working_tape, Block
import ufl

# TODO: After changes to how Expressions are overloaded, read through the docstrings
#       and fix the inaccuracies. Also add docstrings where they are missing.

# Findings:
# _ufl_element, _countedclass, _element are no longer needed.
# _countedclass previously been an attribute of a ufl Counted class, which ufl.Coefficient inherited from.
# _ufl_element being defined before constructor to ufl.Coefficient in Expression.__init__,
# and _element set to be _ufl_element in ufl.Coefficient constructor.

#_IGNORED_EXPRESSION_ATTRIBUTES = ["_ufl_element", "_ufl_shape", "_ufl_function_space", "_count", "_countedclass", "_repr", 
#                                  "_element", "this", "_value_shape", "user_parameters", "_hash", "block_output", "dependencies", "user_defined_derivatives",
#                                  "original_block_output", "tape", "_ad_initialized", "derivative"]

_IGNORED_EXPRESSION_ATTRIBUTES = ["user_defined_derivatives", "dependencies"]

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
            user_init = dict_.pop("__init__", None)

            def __init__(self, *args, **kwargs):
                """Overloaded init method of user-defined Expression classes.

                """
                OverloadedType.__init__(self, *args, **kwargs)
                if user_init is not None:
                    user_init(self, *args, **kwargs)
                Expression.__init__(self, *args, **kwargs)

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

            original_init(self, *args, **kwargs)

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
        OverloadedType.__init__(self, *args, **kwargs)
        original_init(self, cppcode, *args, **kwargs)
        Expression.__init__(self, *args, **kwargs)

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
        annotate_tape = kwargs.pop("annotate_tape", True)
        if annotate_tape:
            self._ad_attributes_dict = {}
            for k in dir(self):
                if k not in _IGNORED_EXPRESSION_ATTRIBUTES:
                    self._ad_attributes_dict[k] = getattr(self, k)
            
            tape = get_working_tape()
            block = ExpressionBlock(self)
            tape.add_block(block)
            block.add_output(self.block_output)
            self.block_output.adj_value = []

    def __setattr__(self, k, v):
        if hasattr(self, "_ad_attributes_dict"):
            if k not in _IGNORED_EXPRESSION_ATTRIBUTES:    
                self.block_output.save_output()
                self._ad_attributes_dict[k] = v
                    
                tape = get_working_tape()
                block = ExpressionBlock(self)
                tape.add_block(block)
                block.add_output(self.create_block_output())
                self.block_output.adj_value = []
        backend.Expression.__setattr__(self, k, v)

    def _ad_copy(self):
        class CopiedExpression(object):
            def __init__(self, expression, attributes):
                self.expression = expression
                self.attributes = attributes.copy()

            def _ad_saved_output(self):
                for k in self.attributes:
                    backend.Expression.__setattr__(self.expression, k, self.attributes[k])
                return self.expression

        return CopiedExpression(self, self._ad_attributes_dict)


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

                interp = backend.interpolate(self.expression.user_defined_derivatives[c], V)
                if isinstance(c, backend.Constant):
                    adj_output = adj_input.inner(interp.vector())
                else:
                    adj_output = adj_input*interp.vector()

                block_output.add_adj_output(adj_output)


# TODO: Would probably be better to have an auto-update file that writes the list.
# Probably not that costly to do this once on import though. 
class _DummyExpressionClass(Expression):
    def eval(self, value, x):
        pass

tmp = _DummyExpressionClass(degree=1, annotate_tape=False)
_IGNORED_EXPRESSION_ATTRIBUTES += dir(tmp)
del tmp
tmp = Expression("1", degree=1, annotate_tape=False)
_IGNORED_EXPRESSION_ATTRIBUTES += dir(tmp)
del tmp



