"""This script outputs the correct ignored expression attributes list for the _IGNORED_EXPRESSION_ATTRIBUTES
defined in fenics_adjoint/types/expression.py.

"""
from fenics import *
from fenics_adjoint import *

ignored_attrs = []

class _DummyExpressionClass(Expression):
    def eval(self, value, x):
        pass

tmp = _DummyExpressionClass(degree=1, annotate_tape=False)
ignored_attrs += dir(tmp)
tmp = Expression("1", degree=1, annotate_tape=False)
ignored_attrs += dir(tmp)

# Attributes added in python3, might not be present on current interpreter,
# so we add them just in case.
ignored_attrs.append("__dir__")
ignored_attrs.append("__init_subclass__")

unique_list = set(ignored_attrs)
print(list(unique_list))
