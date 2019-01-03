from .block_variable import BlockVariable
from .tape import no_annotations


class Placeholder(BlockVariable):
    """A class that functions as a placeholder for block variables (for dependencies).

    This means that you can replace the dependency of a Block with another block variable
    on the fly. Do note that Block outputs can not be placeholders.

    The placeholders are useful when you require earlier values in the computational graph
    to be values computed later in the computational graph from the previous recomputation.
    This can be the case for the initial guess for iterative solvers.
    If the recomputations are expected to only slightly change the solution,
    then using the previously found solution as initial guess can significantly speed up the recomputation.

    Usage:
        >>> u = OverloadedType()
        >>> p = Placeholder(u)
        >>> v = annotated_operator(u)
        >>> p.set_value(v)

        Each recomputation will now use the previously computed v as input
        to annotated_operator.

    """
    def __init__(self, obj):
        super(Placeholder, self).__init__(obj)
        self.block_variable = obj.block_variable
        obj.block_variable = self
        self.linked_bv = None

    def set_value(self, value):
        self.linked_bv = value.block_variable

    @no_annotations
    def save_output(self, overwrite=True):
        pass

    @property
    def saved_output(self):
        if self.linked_bv is not None:
            return self.linked_bv.saved_output
        else:
            return self.block_variable.saved_output

    def will_add_as_dependency(self):
        pass

    def will_add_as_output(self):
        pass
