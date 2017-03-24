from fenics import *
from fenics_adjoint import *


def test_constant():
	mesh = IntervalMesh(10, 0, 1)
	V = FunctionSpace(mesh, "Lagrange", 1)

	c = Constant(1)
	f = Function(V)
	f.vector()[:] = 1

	u = Function(V)
	v = TestFunction(V)
	bc = DirichletBC(V, Constant(1), "on_boundary")

	F = inner(grad(u), grad(v))*dx - f**2*v*dx
	solve(F == 0, u, bc)

	J = Functional(c**2*u*dx)
	Jhat = ReducedFunctional(J, c)
	_test_adjoint_constant(Jhat, c)


def _test_adjoint_constant(J, c):
    import numpy.random

    h = Constant(1)

    eps_ = [0.01/2.0**i for i in range(4)]
    residuals = []
    for eps in eps_:

        Jp = J(Constant(c + eps*h))
        Jm = J(c)

        dJdc = J.derivative()

        residual = abs(Jp - Jm - eps*dJdc)
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    print r

    tol = 1E-1
    assert( r[-1] > 2-tol )


def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))

    return r