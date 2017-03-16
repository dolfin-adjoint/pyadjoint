from fenics import *
from fenics_adjoint import *

# For asserts
OverloadedType = tape.OverloadedType


def test_subclass_expression():
	class MyExpression1(Expression):
	    def eval_cell(self, value, x, ufc_cell):
	        if ufc_cell.index > 10:
	            value[0] = 1.0
	        else:
	            value[0] = -1.0

	f = MyExpression1(degree=1)

	# Assert

	# Expression is removed from bases by the metaclass:
	assert(Expression not in MyExpression1.__bases__)

	# OverloadedType is a base of the subclass:
	assert(OverloadedType in MyExpression1.__bases__)
	assert(isinstance(f, OverloadedType))


def test_jit_expression():
	f = Expression("a*sin(k*pi*x[0])*cos(k*pi*x[1])", a=2, k=3, degree=2)

	# Assert
	assert(isinstance(f, OverloadedType))


def test_jit_expression_evaluations():
	f = Expression("u", u=1, degree=1)

	assert(f.u == 1)
	assert(f(0.0) == 1)

	f.user_parameters['u'] = 2

	assert(f(0.0) == 2)
	assert(f.u == 2)


