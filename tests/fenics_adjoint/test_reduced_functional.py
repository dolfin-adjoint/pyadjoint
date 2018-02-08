import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand
from numpy.testing import assert_approx_equal

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

    J = assemble(c**2*u*dx)
    Jhat = ReducedFunctional(J, Control(c))
    assert(taylor_test(Jhat, c, Constant(1)) > 1.9)


def test_function():
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

    J = assemble(c**2*u*dx)
    Jhat = ReducedFunctional(J, Control(f))
    
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert(taylor_test(Jhat, f, h) > 1.9)


def test_wrt_function_dirichlet_boundary():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10,10)

    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    class Up(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1)

    class Down(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1)

    left = Left()
    right = Right()
    up = Up()
    down = Down()

    boundary = FacetFunction("size_t", mesh)
    boundary.set_all(0)
    up.mark(boundary, 1)
    down.mark(boundary,2)
    ds = Measure("ds", subdomain_data=boundary)

    bc_func = project(Expression("sin(x[1])", degree=1, annotate=False), V, annotate=False)
    bc1 = DirichletBC(V,bc_func,left)
    bc2 = DirichletBC(V,2,right)
    bc = [bc1,bc2]

    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V)
    f.vector()[:] = 10

    a = inner(grad(u), grad(v))*dx
    L = inner(f,v)*dx + inner(g1,v)*ds(1) + inner(g2,v)*ds(2)

    solve(a==L,u_,bc)

    J = assemble(u_**2*dx)

    Jhat = ReducedFunctional(J, Control(bc_func))
    h = Function(V)
    h.vector()[:] = rand(V.dim())

    assert(taylor_test(Jhat, bc_func, h) > 1.9)


def test_time_dependent():
    # Defining the domain, 100 points from 0 to 1
    mesh = IntervalMesh(100, 0, 1)

    # Defining function space, test and trial functions
    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    # Marking the boundaries
    def left(x, on_boundary):
        return near(x[0],0)

    def right(x, on_boundary):
        return near(x[0],1)

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
    control = Control(u_1)

    a = u_1*u*v*dx + dt*f*inner(grad(u),grad(v))*dx
    L = u_1*v*dx

    # Time loop
    t = dt
    while t <= T:
        solve(a == L, u_, bc)
        u_1.assign(u_)
        t += dt

    J = assemble(u_1**2*dx)

    Jhat = ReducedFunctional(J, control)
    
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert(taylor_test(Jhat, control.data(), h) > 1.9)

def test_burgers():
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
    Jhat = ReducedFunctional(J, Control(ic))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert(taylor_test(Jhat, ic, h) > 1.9)

def test_expression():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    bc = DirichletBC(V, Constant(1), "on_boundary")
    a = Function(V)
    a.vector()[:] = 1
    f = Expression("t*a", a=a, t=0.1, degree=1)
    f_deriv = Expression("t", t=0.1, degree=1)
    f.user_defined_derivatives = {a: f_deriv}

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
    Jhat = ReducedFunctional(J, Control(a))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert(taylor_test(Jhat, a, h) > 1.9)

def test_projection():
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
    Jhat = ReducedFunctional(J, Control(k))

    h = Constant(1.0)
    assert(taylor_test(Jhat, k, h) > 1.9)

def test_projection_function():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    bc = DirichletBC(V, Constant(1), "on_boundary")
    g = Function(V)
    g = project(Expression("sin(x[0])*sin(x[1])", degree=1, annotate_tape=False), V, annotate=False)
    expr = Expression("sin(g*x[0])", g=g, degree=1)
    expr.user_defined_derivatives = {g: Expression("x[0]*cos(g*x[0])", g=g, degree=1, annotate=False)}
    f = project(expr, V)

    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    solve(a == L, u_, bc)

    J = assemble(u_**2*dx)
    Jhat = ReducedFunctional(J, Control(g))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert(taylor_test(Jhat, g, h) > 1.9)

def test_assemble_recompute():
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
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert(taylor_test(Jhat, f, h) > 1.9)

def test_solve_output_control():
    set_working_tape(Tape())
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = Function(V)
    t = Constant(1.0)
    u_ = Function(V)
    F = inner(grad(u), grad(v))*dx - (u_ + t)*v*dx
    bc = DirichletBC(V, 1, "on_boundary")

    dt = 0.1
    T = 1.5
    while t.values()[0] <= T:
        if 1.3-1E-08 < t.values()[0] < 1.3+1E-08:
            c = Control(u)
            p_value = u.copy(deepcopy=True)
        solve(F == 0, u, bc)
        u_.assign(u)
        t.assign(t.values()[0] + dt)

    J = assemble(inner(u, u)*dx)
    Jhat = ReducedFunctional(J, c)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(Jhat, p_value, h) > 1.9

def test_multiple_control():
    set_working_tape(Tape())
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = Function(V)
    t = Constant(1.0)
    u_ = Function(V)
    F = inner(grad(u), grad(v))*dx - (u_ + t)*v*dx
    bc = DirichletBC(V, 1, "on_boundary")

    dt = 0.1
    T = 1.5
    while t.values()[0] <= T:
        if 1.3-1E-08 < t.values()[0] < 1.3+1E-08:
            c = Control(u)
            p_value = u.copy(deepcopy=True)
        solve(F == 0, u, bc)
        u_.assign(u)
        t.assign(t.values()[0] + dt)

    # If we don't do annotate=False we have control that depends on the other.
    a = p_value.copy(deepcopy=True, annotate=False)
    c = [c, Control(a)]

    J = assemble(a*a*inner(u, u)*dx)
    Jhat = ReducedFunctional(J, c)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    h2 = Function(V)
    h2.vector()[:] = rand(V.dim())
    assert taylor_test(Jhat, [p_value, p_value], [h, h2]) > 1.9


def test_dirichlet_updating():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = Function(V)
    f = Function(V)
    f.vector()[:] = 1

    t = Constant(0)
    dt = 0.1
    bc = DirichletBC(V, t, "on_boundary")

    T = 0.3
    F = inner(grad(u), grad(v)) * dx - f * v * dx

    J = 0
    while t.values()[0] <= T:
        solve(F == 0, u, bc)
        t.assign(t.values()[0] + dt)
        J += dt*assemble(u**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    h = project(Constant(1), V, annotate=False)
    assert taylor_test(Jhat, f, h) > 1.9


def test_expression_update():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = Function(V)
    t = Constant(0.3)
    a = Constant(2)
    f = Expression("t*t*a*a", t=t, a=a, degree=1)
    f.user_defined_derivatives = {a: Expression("2*t*t*a", t=t, a=a, degree=1, annotate=False)}
    g = Constant(1)

    dt = 0.1
    bc = DirichletBC(V, 1, "on_boundary")

    T = 0.3
    F = inner(grad(u), grad(v)) * dx - g * f * v * dx

    i = 0.0
    j = 0.3
    while i <= j:
        solve(F == 0, u, bc)
        i += dt
        t.assign(t.values()[0] + dt)

    J = assemble(u**2*dx)
    Jhat = ReducedFunctional(J, Control(a))
    h = Constant(1)
    assert taylor_test(Jhat, a, h)


def test_function_split():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V_element = VectorElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, V_element)

    f = project(Expression(("x[0]", "x[1]"), degree=1, annotate=False), V, annotate=False)

    u = TrialFunction(V)
    v = TestFunction(V)

    u_ = Function(V)
    a = inner(u,v)*dx
    L = inner(f,v)*dx
    solve(a == L, u_)

    f1, f2 = f.split()
    J = assemble(f1*inner(u_, u_)*dx)
    Jhat = ReducedFunctional(J, Control(f))
    h = project(Constant((1, 1)), V, annotate=False)
    assert taylor_test(Jhat, f, h)


def test_pow_assemble():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx
    bc = DirichletBC(V, 1, "on_boundary")

    sol = Function(V)
    solve(a == L, sol, bc)

    p = AdjFloat(3)
    J = assemble(sol**2*dx)**p

    Jhat = ReducedFunctional(J, Control(f))
    assert taylor_test(Jhat, f, Constant(1)) > 1.9


def test_multiple_reduced_functionals():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    controls = []
    u = Function(V)
    v = TestFunction(V)

    b = Constant(2)
    controls.append(Control(b))
    F = inner(grad(u), grad(v))*dx - b*v*dx
    bc = DirichletBC(V, b, "on_boundary")
    solve(F == 0, u, bc)

    a = Constant(1)
    f = project(Expression("a*x[0]*x[1]", a=a, degree=1), V)
    controls.append(Control(f))

    F = inner(grad(u), grad(v))*dx - f*v*dx
    solve(F == 0, u, bc)

    J = assemble(inner(u, u)*dx)
    Jhat = ReducedFunctional(J, controls)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    hs = [Constant(1), h]
    assert taylor_test(Jhat, [b, f], hs) > 1.9

    Jhat = ReducedFunctional(J, controls[1])
    assert taylor_test(Jhat, f, h) > 1.9

    Jhat = ReducedFunctional(J, controls[0])
    assert taylor_test(Jhat, b, Constant(1)) > 1.9


def test_dependent_controls():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.vector()[:] = 1

    c = assemble(f**2*dx)
    controls = [Control(f), Control(c)]

    J = c*assemble(f*dx)

    Jhat = ReducedFunctional(J, controls)

    with pytest.raises(RuntimeError):
        Jhat.optimize()


def test_control_optimized_reduced_functional():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = Function(V)
    t = Constant(1.0)
    u_ = Function(V)
    F = inner(grad(u), grad(v))*dx - (u_ + t)*v*dx
    bc = DirichletBC(V, 1, "on_boundary")

    dt = 0.1
    T = 1.5
    while t.values()[0] <= T:
        if 1.3-1E-08 < t.values()[0] < 1.3+1E-08:
            c = Control(u)
            p_value = u.copy(deepcopy=True)
        solve(F == 0, u, bc)
        u_.assign(u)
        t.assign(t.values()[0] + dt)

    J = assemble(inner(u, u)*dx)
    Jhat = ReducedFunctional(J, c)
    h = Function(V)
    h.vector()[:] = rand(V.dim())

    tape = get_working_tape()
    pre_optimized_len = len(tape.get_blocks())
    Jhat.optimize()
    assert len(tape.get_blocks()) < pre_optimized_len
    assert taylor_test(Jhat, p_value, h) > 1.9


def test_functional_optimized_reduced_functional():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.vector()[:] = 1
    control = Control(f)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v)*dx
    L = f*v*dx
    sol = Function(V)
    solve(a == L, sol)

    J1 = assemble(sol**2*dx)
    Jhat = ReducedFunctional(J1, control)

    tape = get_working_tape()
    pre_len = len(tape.get_blocks())
    Jhat.optimize()
    assert pre_len == len(tape.get_blocks())

    J2 = assemble(sol**4*dx)
    assert pre_len < len(tape.get_blocks())
    Jhat.optimize()
    assert pre_len == len(tape.get_blocks())

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(Jhat, f, h) > 1.9


def test_multiple_optimized_reduced_functionals():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.vector()[:] = 1
    control = Control(f)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v)*dx
    L = f*v*dx
    sol = Function(V)
    solve(a == L, sol)

    J1 = assemble(sol**2*dx)
    J2 = assemble(sol**4*dx)
    Jhat = ReducedFunctional(J1, control)

    tape = get_working_tape()
    tape2 = tape.copy()
    pre_len = len(tape.get_blocks())
    assert pre_len == len(tape.get_blocks())
    assert pre_len == len(tape2.get_blocks())

    Jhat.optimize()
    assert pre_len > len(tape.get_blocks())
    assert pre_len == len(tape2.get_blocks())

    Jhat2 = ReducedFunctional(J2, control, tape=tape2)
    Jhat2.optimize()

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(Jhat, f, h) > 1.9
    assert taylor_test(Jhat2, f, h) > 1.9


def test_recompute_expression_bc():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.vector()[:] = 1
    u = TrialFunction(V)
    v = TestFunction(V)

    sol = Function(V)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    c = Constant(1)
    exp = Expression("c", c=c, degree=1)
    bc = DirichletBC(V, exp, "on_boundary")
    solve(a == L, sol, bc)

    J = assemble(sol*sol*dx)
    m = Control(f)
    Jhat = ReducedFunctional(J, m)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(Jhat, f, h) > 1.9


def test_wrong_number_of_values():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)
    f = Constant(1.0)
    g = Constant(2.0)
    h = AdjFloat(3.0)

    J = h*assemble(f*g*dx(domain=mesh))
    Jhat = ReducedFunctional(J, [Control(f), Control(g), Control(h)])

    with pytest.raises(ValueError):
        Jhat(Constant(1.0))

    with pytest.raises(ValueError):
        Jhat([Constant(1.0), Constant(2.0)])

    assert_approx_equal(Jhat([Constant(1.0), Constant(2.0), AdjFloat(3.0)]), J)


def test_eval_callback():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)
    f = Constant(1.0)

    u = TrialFunction(V)
    v = TestFunction(V)

    sol = Function(V)
    a = u*v*dx
    L = f*v*dx
    solve(a == L, sol)

    J = assemble(sol**2*dx)

    calls_counter = {
        "eval_cb_pre_calls": 0,
        "eval_cb_post_calls": 0,
        "derivative_cb_pre_calls": 0,
        "derivative_cb_post_calls": 0
    }

    def eval_cb_pre(values):
        calls_counter["eval_cb_pre_calls"] += 1

    def eval_cb_post(functional_value, values):
        calls_counter["eval_cb_post_calls"] += 1
        assert calls_counter["eval_cb_pre_calls"] == calls_counter["eval_cb_post_calls"]

    def derivative_cb_pre(values):
        calls_counter["derivative_cb_pre_calls"] += 1

    def derivative_cb_post(functional_value, derivatives, values):
        calls_counter["derivative_cb_post_calls"] += 1
        assert calls_counter["derivative_cb_post_calls"] == calls_counter["derivative_cb_pre_calls"]

    Jhat = ReducedFunctional(J, Control(f),
                             eval_cb_pre=eval_cb_pre,
                             eval_cb_post=eval_cb_post,
                             derivative_cb_pre=derivative_cb_pre,
                             derivative_cb_post=derivative_cb_post)

    Jhat(Constant(2.0))
    assert calls_counter["derivative_cb_post_calls"] == calls_counter["derivative_cb_pre_calls"]
    assert calls_counter["derivative_cb_post_calls"] == 0
    Jhat.derivative()
    assert calls_counter["eval_cb_pre_calls"] == 1
    assert calls_counter["derivative_cb_pre_calls"] == 1

