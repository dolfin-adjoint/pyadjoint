from firedrake import *
from firedrake_adjoint import *

def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))

    return r


def taylor_adjoint(J, f):
    import numpy.random
    tape = Tape()
    set_working_tape(tape)

    V = f.function_space()
    h = Function(V)
    for i, e in enumerate(V._components):
        h.sub(i).vector()[:] = numpy.random.rand(*[e.node_count, *e.shape])

    eps_ = [0.01/2.0**i for i in range(5)]
    residuals = []
    for eps in eps_:

        Jp = J(f + eps*h)
        tape.clear_tape()
        Jm = J(f)
        Jm.adj_value = 1.0
        tape.evaluate_adj()

        dJdf = f.adj_value

        residual = abs(Jp - Jm - eps*dJdf.inner(h.vector()))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    print(r)
    print(residuals)

    tol = 1E-1
    assert( r[-1] > 2-tol )
