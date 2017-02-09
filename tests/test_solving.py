from fenics import *
from fenics_adjoint import *

def test_linear_problem():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 1

    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    def J(f):
        a = inner(grad(u), grad(v))*dx
        L = f*v*dx
        solve(a == L, u_, bc)
        return assemble(u_**2*dx)

    _test_adjoint(J, f)

def test_nonlinear_problem():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    def J(f):
        a = inner(grad(u), grad(v))*dx + u**2*v*dx - f*v*dx
        L = 0
        solve(a == L, u, bc)
        return assemble(u**2*dx)

    _test_adjoint(J, f)



def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))

    return r

def _test_adjoint(J, f):
    import numpy.random
    tape = Tape()
    set_working_tape(tape)

    V = f.function_space()
    h = Function(V)
    h.vector()[:] = numpy.random.rand(V.dim())

    eps_ = [0.4/2.0**i for i in range(4)]
    residuals = []
    for eps in eps_:

        Jp = J(f + eps*h)
        tape.clear_tape()
        Jm = J(f)
        Jm.set_initial_adj_input(1.0)
        tape.evaluate()

        dJdf = f.get_adj_output()

        residual = abs(Jp - Jm - eps*dJdf.inner(h.vector()))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    assert( r[-1] > 2-1E-3 )