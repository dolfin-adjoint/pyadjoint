import backend
from pyadjoint.tape import OverloadedType


_backend_ExpressionMetaClass = backend.functions.expression.ExpressionMetaClass
class OverloadedExpressionMetaClass(_backend_ExpressionMetaClass):
    """Overloads the ExpressionMetaClass so that we can create overloaded 
    Expression types.

    """
    def __new__(mcs, class_name, bases, dict_):
        """Handles creation of new Expression classes.

        """
        if class_name == "Expression" or class_name == "CompiledExpression":
            if len(bases) >= 1 and bases[0] == OverloadedType:
                # If this is our own overloaded Expression/CompiledExpression, then we only want to use
                # our own definition, and not call ExpressionMetaClass.__new__.
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

            dict_["__init__"] = __init__

        bases = tuple(bases)

        # Pass everything along to the backend metaclass.
        # This should return a new user-defined Expression 
        # subclass that inherit from OverloadedType.
        original = _backend_ExpressionMetaClass.__new__(mcs, class_name, bases, dict_)

        return original

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
    bases = (OverloadedType, type(original))

    def __init__(self, cppcode, *args, **kwargs):
        """Init method of our overloaded CompiledExpression classes.

        """
        OverloadedType.__init__(self, *args, **kwargs)
        type(original).__init__(self, cppcode, *args, **kwargs)

    return type.__new__(OverloadedExpressionMetaClass, "CompiledExpression", bases, {"__init__": __init__})


class Expression(OverloadedType, backend.Expression):
    """Overloaded Expression class.

    This class has only two functions:
        1. It overloads Expression.__new__ to create overloaded CompiledExpressions.
        
        2. It assigns an overloaded metaclass to all user-defined subclasses of Expression.

    Note: At any given point there are no classes that have this class as a base.

    """
    __metaclass__ = OverloadedExpressionMetaClass
    def __new__(cls, cppcode=None, *args, **kwargs):
        original = backend.Expression.__new__(cls, cppcode, *args, **kwargs)
        return object.__new__(create_compiled_expression(original, cppcode, *args, **kwargs))

