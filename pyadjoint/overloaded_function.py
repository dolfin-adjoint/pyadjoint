from .tape import annotate_tape, get_working_tape, stop_annotating
from .overloaded_type import create_overloaded_object
from .enlisting import Enlist


def overload_function(func, block_class):
    """Create an overloaded version of a function.

    This method makes several assumptions:
    1) The function is explicit, i.e. y = func(x), where y is the output of the operation.
    2) All of y is possible to convert to an OverloadedType.
    3) Unless annotation is turned off, the operation should always be annotated when calling the overloaded function.

    Args:
        func (function): The target function for which to create an overloaded version.
        block_class (type): The Block-subclass that corresponds to `func`.

    Returns:
        function: An overloaded version of `func`

    """
    def _overloaded_function(*args, **kwargs):
        annotate = annotate_tape(kwargs)
        if annotate:
            b_kwargs = block_class.pop_kwargs(kwargs)
            b_kwargs.update(kwargs)
            block = block_class(*args, **b_kwargs)

        with stop_annotating():
            func_output = func(*args, **kwargs)

        r = []
        func_output = Enlist(func_output)
        for out in func_output:
            r.append(create_overloaded_object(out))

        if annotate:
            for out in r:
                block.add_output(out.create_block_variable())
            tape = get_working_tape()
            tape.add_block(block)

        return func_output.delist(r)
    return _overloaded_function


def overloaded_function(block_class):
    """Returns a decorator for functions that should be overloaded

    Args:
        block_class (type): The Block-subclass that corresponds to the function being decorated.

    """
    def decorator(func):
        return overload_function(func, block_class)
    return decorator
