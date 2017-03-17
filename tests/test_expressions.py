from fenics import *
from fenics_adjoint import *

# For asserts
from pyadjoint.tape import OverloadedType

def test_subclass_expression():
	class MyExpression1(Expression):
	    def eval_cell(self, value, x, ufc_cell):
	        if ufc_cell.index > 10:
	            value[0] = 1.0
	        else:
	            value[0] = -1.0

	f = MyExpression1(degree=1)

	# Expression is removed from bases by the metaclass:
	assert(Expression not in MyExpression1.__bases__)

	# OverloadedType is a base of the subclass:
	assert(OverloadedType in MyExpression1.__bases__)
	assert(isinstance(f, OverloadedType))


def test_jit_expression():
	f = Expression("a*sin(k*pi*x[0])*cos(k*pi*x[1])", a=2, k=3, degree=2)

	assert(isinstance(f, OverloadedType))


def test_jit_expression_evaluations():
	f = Expression("u", u=1, degree=1)

	assert(f.u == 1)
	assert(f(0.0) == 1)

	f.user_parameters['u'] = 2

	assert(f(0.0) == 2)
	assert(f.u == 2)


def test_jit_expression_adj():
	mesh = IntervalMesh(10, 0, 1)
	f = Expression("sin(a*x[0])", a=2, degree=1)

	form = f**2*dx(domain=mesh)
	J = assemble(form)
	
	J.set_initial_adj_input(1.0)
	tape = get_working_tape()
	tape.evaluate()

	exp_dJdf = f.get_adj_output()
	
	tape = Tape()
	set_working_tape(tape)

	V = FunctionSpace(mesh, "Lagrange", 1)

	# TODO: Workaround before interpolate is overloaded
	tmp = interpolate(f, V)
	f = Function(V)
	f.vector()[:] = tmp.vector()[:]

	form = f**2*dx
	J = assemble(form)

	J.set_initial_adj_input(1.0)
	tape.evaluate()

	func_dJdf = f.get_adj_output()

	assert((exp_dJdf == func_dJdf).all())


