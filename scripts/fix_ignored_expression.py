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


print(set(ignored_attrs))