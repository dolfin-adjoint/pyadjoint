import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake_adjoint import *


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
    Jhat = ReducedFunctional(J, c)
    assert taylor_test(Jhat, Constant(5), Constant(1)) > 1.9


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
    Jhat = ReducedFunctional(J, f)
    
    h = Function(V)
    from numpy.random import rand
    h.vector()[:] = rand(V.dof_dset.size)
    # Note that if you use f directly, it will not work
    # as expected since f is the control and thus the initial point in control
    # space is changed as you do the test. (Since f.vector is also assigned new values on pertubations)
    g = f.copy(deepcopy=True)

    assert taylor_test(Jhat, g, h) > 1.9


@pytest.mark.parametrize("control", ["dirichlet", "neumann"])
def test_wrt_function_dirichlet_boundary(control):
    mesh = UnitSquareMesh(10,10)

    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc_func = project(sin(y), V)
    bc1 = DirichletBC(V, bc_func, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1,bc2]

    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V)
    f.vector()[:] = 10

    a = inner(grad(u), grad(v))*dx
    L = inner(f,v)*dx + inner(g1,v)*ds(4) + inner(g2,v)*ds(3)

    solve(a==L,u_,bc)

    J = assemble(u_**2*dx)

    if control == "dirichlet":
        Jhat = ReducedFunctional(J, bc_func)
        g = bc_func.copy(deepcopy=True)
        h = Function(V)
        h.vector()[:] = 1
    else:
        Jhat = ReducedFunctional(J, g1)
        g = Constant(2)
        h = Constant(1)

    assert taylor_test(Jhat, g, h) > 1.9


def test_time_dependent():
    # Defining the domain, 100 points from 0 to 1
    mesh = IntervalMesh(100, 0, 1)

    # Defining function space, test and trial functions
    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    # Dirichlet boundary conditions
    bc_left = DirichletBC(V, 1, 1)
    bc_right = DirichletBC(V, 2, 2)
    bc = [bc_left, bc_right]

    # Some variables
    T = 0.5
    dt = 0.1
    f = Function(V)
    f.vector()[:] = 1

    u_1 = Function(V)
    u_1.vector()[:] = 1 

    a = u_1*u*v*dx + dt*f*inner(grad(u),grad(v))*dx
    L = u_1*v*dx

    # Time loop
    t = dt
    while t <= T:
        solve(a == L, u_, bc)
        u_1.assign(u_)
        t += dt

    J = assemble(u_1**2*dx)

    Jhat = ReducedFunctional(J, u_1)
    
    h = Function(V)
    h.vector()[:] = 1
    g = f.copy(deepcopy=True)
    assert taylor_test(Jhat, g, h) > 1.9


def test_mixed_boundary():
    mesh = UnitSquareMesh(10,10)

    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    bc1 = DirichletBC(V, y**2, 1)
    bc2 = DirichletBC(V, 2, 2)
    bc = [bc1,bc2]
    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V)
    f.vector()[:] = 10

    a = f*inner(grad(u), grad(v))*dx
    L = inner(f,v)*dx + inner(g1,v)*ds(4) + inner(g2,v)*ds(3)

    solve(a==L,u_,bc)

    J = assemble(u_**2*dx)

    Jhat = ReducedFunctional(J, f)
    g = Function(f)
    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, g, h) > 1.9


@pytest.mark.xfail(reason="Expression not implemented yet")
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
    Jhat = ReducedFunctional(J, a)

    h = Function(V)
    h.vector()[:] = 1
    g = a.copy(deepcopy=True)
    assert taylor_test(Jhat, g, h) > 1.9


@pytest.mark.xfail(reason="Expression not implemented yet")
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
    Jhat = ReducedFunctional(J, k)

    m = Constant(2.0)
    h = Constant(1.0)
    assert taylor_test(Jhat, m, h) > 1.9


@pytest.mark.xfail(reason="Expression not implemented yet")
def test_projection_function():
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
    Jhat = ReducedFunctional(J, g)

    m = g.copy(deepcopy=True)
    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, m, h) > 1.9


@pytest.mark.xfail(reason="Constant annotation not yet quite right")
def test_assemble_recompute():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    v = TestFunction(V)
    u = Function(V)
    u.vector()[:] = 1

    bc = DirichletBC(V, Constant(1), "on_boundary")
    f = Function(V)
    f.vector()[:] = 2
    expr = Constant(assemble(f**2*dx))
    J = assemble(expr**2*dx(domain=mesh))
    Jhat = ReducedFunctional(J, f)

    m = f.copy(deepcopy=True)
    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(Jhat, m, h) > 1.9
