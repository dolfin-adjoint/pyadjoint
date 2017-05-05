from fenics import *
from fenics_adjoint import *

from numpy.random import rand

# Tolerance in the tests.
tol = 1E-10


def test_tlm_assemble():
    tape = Tape()
    set_working_tape(tape)
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    f = Function(V)
    f.vector()[:] = 5

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    u_ = Function(V)

    bc = DirichletBC(V, 1, "on_boundary")

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    f.set_initial_tlm_input(h.vector())
    tape.evaluate_tlm()
    J.set_initial_adj_input(1.0)
    tape.evaluate()

    assert abs(f.original_block_output.adj_value.inner(h.vector()) - J.block_output.tlm_value) < tol


def test_tlm_bc():
    tape = Tape()
    set_working_tape(tape)
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Constant(1)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, c, "on_boundary")

    F = inner(grad(u), grad(v)) * dx - f ** 2 * v * dx
    solve(F == 0, u, bc)

    J = assemble(c ** 2 * u * dx)

    c.set_initial_tlm_input(Constant(1))
    tape.evaluate_tlm()
    J.set_initial_adj_input(1.0)
    tape.evaluate()

    assert abs(c.original_block_output.adj_value - J.block_output.tlm_value) < tol


def test_time_dependent():
    tape = Tape()
    set_working_tape(tape)
    # Defining the domain, 100 points from 0 to 1
    mesh = IntervalMesh(100, 0, 1)

    # Defining function space, test and trial functions
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    # Marking the boundaries
    def left(x, on_boundary):
        return near(x[0], 0)

    def right(x, on_boundary):
        return near(x[0], 1)

    # Dirichlet boundary conditions
    bc_left = DirichletBC(V, 1, left)
    bc_right = DirichletBC(V, 2, right)
    bc = [bc_left, bc_right]

    # Some variables
    T = 0.5
    dt = 0.1
    f = Function(V)
    f.vector()[:] = 1

    u_1 = Function(V)
    u_1.vector()[:] = 1

    a = u_1 * u * v * dx + dt * f * inner(grad(u), grad(v)) * dx
    L = u_1 * v * dx

    # Time loop
    t = dt
    while t <= T:
        solve(a == L, u_, bc)
        u_1.assign(u_)
        t += dt

    J = assemble(u_1 ** 2 * dx)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    u_1.set_initial_tlm_input(h.vector())
    tape.evaluate_tlm()
    J.set_initial_adj_input(1.0)
    tape.evaluate()

    assert abs(u_1.original_block_output.adj_value.inner(h.vector()) - J.block_output.tlm_value) < tol


def test_burgers():
    tape = Tape()
    set_working_tape(tape)
    n = 30
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 2)

    def Dt(u, u_, timestep):
        return (u - u_)/timestep

    pr = project(Expression("sin(2*pi*x[0])", degree=1), V)
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

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    ic.set_initial_tlm_input(h.vector())
    tape.evaluate_tlm()
    J.set_initial_adj_input(1.0)
    tape.evaluate()

    assert abs(ic.original_block_output.adj_value.inner(h.vector()) - J.block_output.tlm_value) < tol


def test_expression():
    tape = Tape()
    set_working_tape(tape)
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    a = Function(V)
    a.vector()[:] = 1
    f = Expression("t*a", a=a, t=0.1, degree=1)
    f_deriv = Expression("t", t=0.1, degree=1)
    f.user_defined_derivatives = {a: f_deriv}
    c = Function(V)
    c.vector()[:] = 1
    bc = DirichletBC(V, c, "on_boundary")

    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v))*dx - f*v*dx

    t = 0.1
    dt = 0.1
    T = 0.3
    while t <= T:
        solve(F == 0, u, bc)
        t += dt
        f.t = t

    J = assemble(u**2*dx)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    a.set_initial_tlm_input(h.vector())
    tape.evaluate_tlm()
    J.set_initial_adj_input(1.0)
    tape.evaluate()

    assert abs(a.original_block_output.adj_value.inner(h.vector()) - J.block_output.tlm_value) < tol


def test_projection():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    bc = DirichletBC(V, Constant(1), "on_boundary")
    k = Constant(2.0)
    expr = Expression("sin(k*x[0])", k=k, degree=1)
    expr.user_defined_derivatives = {k: Expression("x[0]*cos(k*x[0])", k=k, degree=1, annotate_tape=False)}
    f = project(expr, V)

    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)

    k.set_initial_tlm_input(Constant(1))
    tape.evaluate_tlm()
    J.set_initial_adj_input(1.0)
    tape.evaluate()

    assert abs(k.original_block_output.adj_value - J.block_output.tlm_value) < tol


def test_projection_function():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    bc = DirichletBC(V, Constant(1), "on_boundary")
    g = Function(V)
    g = project(Expression("sin(x[0])*sin(x[1])", degree=1, annotate_tape=False), V, annotate_tape=False)
    expr = Expression("sin(g*x[0])", g=g, degree=1)
    expr.user_defined_derivatives = {g: Expression("x[0]*cos(g*x[0])", g=g, degree=1, annotate_tape=False)}
    f = project(expr, V)

    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    g.set_initial_tlm_input(h.vector())
    tape.evaluate_tlm()
    J.set_initial_adj_input(1.0)
    tape.evaluate()

    assert abs(g.original_block_output.adj_value.inner(h.vector()) - J.block_output.tlm_value) < 1E-4


def test_assemble_recompute():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = Function(V)
    u.vector()[:] = 1

    bc = DirichletBC(V, Constant(1), "on_boundary")
    f = Function(V)
    f.vector()[:] = 2
    k = assemble(f**2*dx)
    expr = Expression("k", k=k, degree=1)
    expr.user_defined_derivatives = {k: Expression("1", degree=1, annotate_tape=False)}
    J = assemble(expr**2*dx(domain=mesh))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    f.set_initial_tlm_input(h.vector())
    tape.evaluate_tlm()
    J.set_initial_adj_input(1.0)
    tape.evaluate()

    assert abs(f.original_block_output.adj_value.inner(h.vector()) - J.block_output.tlm_value) < tol




