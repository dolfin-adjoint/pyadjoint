import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand

def test_simple_solve():
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
    c = Control(f)
    Jhat = ReducedFunctional(J, c)

    h = Function(V)
    h.vector()[:] = rand(V.dim())

    dJdm = Jhat.derivative().vector().inner(h.vector())
    H = Hessian(J, c)
    Hm = H(h).vector().inner(h.vector())
    assert(taylor_test(Jhat, f, h, dJdm=dJdm, Hm=Hm) > 2.9)


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
    J.adj_value = 1.0
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    f.tlm_value = h
    g.tlm_value = h

    tape.evaluate()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    dJdm = J.block_variable.tlm_value
    Hm = f.original_block_variable.hessian_value.inner(h.vector()) + g.original_block_variable.hessian_value.inner(h.vector())

    m_1 = f.copy(deepcopy=True)
    m_2 = g.copy(deepcopy=True)

    assert(conv_mixed(J, f, g, m_1, m_2, h, h, dJdm, Hm) > 2.9)


def test_function():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 2)

    c = Constant(4)
    f = Function(V)
    f.vector()[:] = 3

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v)) * dx + u**2*v*dx - f ** 2 * v * dx - c**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c ** 2 * u ** 2 * dx)
    Jhat = ReducedFunctional(J, f)

    h = Function(V)
    h.vector()[4] = 1

    J.adj_value = 1.0
    f.tlm_value = h
    c.tlm_value = Constant(1)

    tape.evaluate()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    g = f.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value
    Hm = f.original_block_variable.hessian_value.inner(h.vector()) + c.original_block_variable.hessian_value[0]

    assert(conv_mixed(J, f, c, g, Constant(4), h, Constant(1), dJdm=dJdm, Hm=Hm) > 2.9)


def test_nonlinear():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 5

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - u**2*v*dx - f * v * dx
    solve(F == 0, u, bc)

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())

    J.adj_value = 1.0
    f.tlm_value = h

    tape.evaluate()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    g = f.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value
    Hm = f.original_block_variable.hessian_value.inner(h.vector())
    assert(taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9)

def test_dirichlet():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 30

    u = Function(V)
    v = TestFunction(V)
    c = Function(V)
    c.vector()[:] = 1
    bc = DirichletBC(V, c, "on_boundary")

    F = inner(grad(u), grad(v)) * dx + u**4*v*dx - f**2 * v * dx
    solve(F == 0, u, bc)

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(c))

    h = Function(V)
    h.vector()[:] = rand(V.dim())

    J.adj_value = 1.0
    c.tlm_value = h

    tape.evaluate()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    g = c.copy(deepcopy=True)

    dJdm = J.block_variable.tlm_value

    Hm = c.original_block_variable.hessian_value.inner(h.vector())
    assert(taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9)

def test_expression():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "Lagrange", 1)

    u = Function(V)
    v = TestFunction(V)
    c = Constant(1)
    f = Expression("c*c*c", c=c, degree=1)
    first_deriv = Expression("3*c*c", c=c, degree=1, annotate=False)
    second_deriv = Expression("6*c", c=c, degree=1, annotate=False)
    f.user_defined_derivatives = {c: first_deriv}
    first_deriv.user_defined_derivatives = {c: second_deriv}
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - f * v * dx
    solve(F == 0, u, bc)

    J = assemble(u ** 4 * dx)
    Jhat = ReducedFunctional(J, Control(c))

    h = Constant(1)

    J.adj_value = 1.0
    c.tlm_value = h

    tape.evaluate()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    g = Constant(1)

    dJdm = J.block_variable.tlm_value

    Hm = c.original_block_variable.hessian_value
    assert(taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9)


def test_burgers():
    tape = Tape()
    set_working_tape(tape)
    n = 30
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 2)

    def Dt(u, u_, timestep):
        return (u - u_)/timestep

    pr = project(Expression("sin(2*pi*x[0])", degree=1, annotate=False), V, annotate=False)
    ic = Function(V)
    ic.vector()[:] = pr.vector()[:]

    u_ = Function(V)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, ic, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    solve(F == 0, u, bc)
    u_.assign(u)
    t += float(timestep)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx

    end = 0.2
    while (t <= end):
        solve(F == 0, u, bc)
        u_.assign(u)

        t += float(timestep)

    J = assemble(u_*u_*dx + ic*ic*dx)

    Jhat = ReducedFunctional(J, Control(ic))
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    g = ic.copy(deepcopy=True)
    J.adj_value = 1.0
    ic.tlm_value = h
    tape.evaluate()
    tape.evaluate_tlm()

    J.block_variable.hessian_value = 0
    tape.evaluate_hessian()

    dJdm = J.block_variable.tlm_value
    Hm = ic.original_block_variable.hessian_value.inner(h.vector())
    assert(taylor_test(Jhat, g, h, dJdm=dJdm, Hm=Hm) > 2.9)

# Temporary mixed controls taylor test until pyadjoint natively supports it.
def conv_mixed(J, f, g, m_1, m_2, h_1, h_2, dJdm, Hm):
    tape = get_working_tape()
    def J_eval(m_1, m_2):
        f.adj_update_value(m_1)
        g.adj_update_value(m_2)

        blocks = tape.get_blocks()
        for i in range(len(blocks)):
            blocks[i].recompute()

        return J.block_variable.saved_output

    Jm = J_eval(m_1, m_2)

    residuals = []
    epsilons = [0.01 / 2 ** i for i in range(4)]
    for eps in epsilons:
        perturbation_1 = h_1._ad_mul(eps)
        perturbation_2 = h_2._ad_mul(eps)
        Jp = J_eval(m_1._ad_add(perturbation_1), m_2._ad_add(perturbation_2))

        res = abs(Jp - Jm - eps * dJdm - 0.5 * eps ** 2 * Hm)
        residuals.append(res)
    print(residuals)
    return min(convergence_rates(residuals, epsilons))


def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i] / E_values[i - 1]) / log(eps_values[i] / eps_values[i - 1]))
    print(r)
    return r

