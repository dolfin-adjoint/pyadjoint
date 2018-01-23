import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

def test_projection():
	tape = Tape()
	set_working_tape(tape)

	mesh = IntervalMesh(10, 0, 1)
	V = FunctionSpace(mesh, "Lagrange", 1)

	class MyExpression1(UserExpression):
	    def eval_cell(self, value, x, ufc_cell):
	        if ufc_cell.index > 10:
	            value[0] = 1.0
	        else:
	            value[0] = -1.0

	f = MyExpression1(degree=1)

	u = project(f, V)
	print(type(u))

	J = assemble(u**2*dx)

	J.adj_value = 1.0
	tape.evaluate()

	dJdf = f.adj_value
	# TODO: This test does nothing. Make it actually test projection.
	#print(dJdf.array())
