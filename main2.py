from fenics import *
from src import *

mesh = IntervalMesh(10, 0, 1)
V = FunctionSpace(mesh, "Lagrange", 1)

u = Function(V)
#u.vector()[:] = 1
v = TestFunction(V)
bc = DirichletBC(V, Constant(0), "on_boundary")

f = Function(V)
f.vector()[:] = 1

eq = inner(grad(u), grad(v))*dx + inner(grad(u), grad(f))*v*dx - f*v*dx
solve(eq == 0, u, bc)

j = assemble(u**2*dx)
j.set_initial_adj_input(1.0)
tape = Tape.get_tape()

tape.evaluate()

dJdf = f.get_adj_output()


def fwd(f):
	eq = inner(grad(u), grad(v))*dx + inner(grad(u), grad(f))*v*dx - f*v*dx
	solve(eq == 0, u, bc)
	return assemble(u**2*dx)

def convergence_rate(E_values, eps_values):
	from numpy import log
	r = []
	for i in range(1, len(eps_values)):
		r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))

	return r

def test_adj():
	tape = Tape.get_tape()
	J = fwd

	eps_ = [0.4/2.0**i for i in range(4)]
	residuals = []
	for eps in eps_:

		h = Function(V)
		h.vector()[:] = 1

		Jp = J(f + eps*h)
		#tape.clear_tape()
		Jm = J(f)
		
		print dJdf.array()
		#print (Jp - Jm)

		residual = abs(Jp - Jm - eps*dJdf.inner(h.vector()))
		residuals.append(residual)

	r = convergence_rate(residuals, eps_)
	print r

test_adj()

