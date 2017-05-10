from fenics import *
from fenics_adjoint import *

def _test_simple_solve():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 2

    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = f*v*dx

    u_ = Function(V)

    solve(a == L, u_)

    L = u_*v*dx

    u_sol = Function(V)
    solve(a == L, u_sol)

    J = assemble(u_sol**4*dx)
    Jhat = ReducedFunctional(J, f)

    tape = get_working_tape()

    h = Function(V)
    h.vector()[:] = 1
    f.set_initial_tlm_input(h)
    J.set_initial_adj_input(1.0)

    tape.evaluate()
    tape.evaluate_tlm()
    J.block_output.hessian_value = 0

    tape.evaluate_hessian()

    m = f.copy(deepcopy=True)
    dJdm = f.original_block_output.adj_value.inner(h.vector())
    Hm = f.original_block_output.hessian_value.inner(h.vector())
    assert(taylor_test(Jhat, m, h, dJdm=dJdm, Hm=Hm) > 2.9)


def test_mixed_derivatives():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 2

    g = Function(V)
    g.vector()[:] = 3

    u = TrialFunction(V)
    v = TestFunction(V)

    a = f**2*u*v*dx
    L = g**2*v*dx

    u_ = Function(V)
    solve(a == L, u_)

    J = assemble(u_**2*dx)
    J.set_initial_adj_input(1.0)
    h = Function(V)
    h.vector()[:] = 1
    f.set_initial_tlm_input(h)
    g.set_initial_tlm_input(h)

    tape.evaluate()
    tape.evaluate_tlm()

    J.block_output.hessian_value = 0
    tape.evaluate_hessian()

    dJdm = J.block_output.tlm_value
    Hm = f.original_block_output.hessian_value.inner(h.vector()) + g.original_block_output.hessian_value.inner(h.vector())

    m_1 = f.copy(deepcopy=True)
    m_2 = g.copy(deepcopy=True)

    assert(conv_mixed(J, f, g, m_1, m_2, h, dJdm, Hm) > 2.9)


def conv_mixed(J, f, g, m_1, m_2, h, dJdm, Hm):
    tape = get_working_tape()
    def J_eval(m_1, m_2):
        f.adj_update_value(m_1)
        g.adj_update_value(m_2)

        blocks = tape.get_blocks()
        for i in range(len(blocks)):
            blocks[i].recompute()

        return J.block_output.get_saved_output()

    Jm = J_eval(m_1, m_2)

    residuals = []
    epsilons = [0.01 / 2 ** i for i in range(4)]
    for eps in epsilons:
        perturbation = h._ad_mul(eps)
        Jp = J_eval(m_1._ad_add(perturbation), m_2._ad_add(perturbation))

        res = abs(Jp - Jm - eps * dJdm - 0.5 * eps ** 2 * Hm)
        residuals.append(res)

    return min(convergence_rates(residuals, epsilons))


def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i] / E_values[i - 1]) / log(eps_values[i] / eps_values[i - 1]))
    print r
    return r
