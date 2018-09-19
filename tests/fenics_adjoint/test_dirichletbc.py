import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def test_simple_expression():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    m = Constant(2.0)
    expr = Expression("m*m*m", m=m, degree=1)
    deriv = Expression("3*m*m", m=m, degree=1)
    deriv.user_defined_derivatives = {m: Expression("6*m", m=m, degree=1)}
    expr.user_defined_derivatives = {m: deriv}

    bc = DirichletBC(V, expr, "on_boundary")
    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)
    a = inner(grad(u), grad(v))*dx
    L = Constant(0)*v*dx

    solve(a == L, sol, bc)
    J = assemble(sol**2*dx)

    Jhat = ReducedFunctional(J, Control(m))

    h = Constant(0.1)
    dJdm = Jhat.derivative()._ad_dot(h)
    Hm = Jhat.hessian(h)._ad_dot(h)
    assert taylor_test(Jhat, m, h, dJdm=dJdm, Hm=Hm) > 2.9


def test_expression_constant():
    mesh = UnitSquareMesh(3, 3)
    V_h = VectorElement("CG", mesh.ufl_cell(), 2)
    Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V_h * Q_h)

    m = Constant(2.0)
    expr = Expression(("m*m*(x[1]-0.5)*(x[1]-0.5)", "0.3*m*m"), m=m, degree=1)
    expr.user_defined_derivatives = {m: Expression(("2*m*(x[1]-0.5)*(x[1]-0.5)", "0.6*m"), m=m, degree=1)}

    noslip = DirichletBC(W.sub(0), (0, 0),
                         "on_boundary && (x[1] >= 0.9 || x[1] < 0.1)")
    inflow = DirichletBC(W.sub(0), expr, "on_boundary && x[0] <= 0.1 && x[1] < 0.9 && x[1] > 0.1")

    bcs = [inflow, noslip]
    s = Function(W, name="State")
    v, q = TestFunctions(W)
    x = TrialFunction(W)
    u, p = split(x)

    nu = Constant(1)
    a = (nu * inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         - inner(q, div(u)) * dx
         )
    L = inner(Constant((0.0, 0.0)), v) * dx

    A, b = assemble_system(a, L, bcs)
    solve(A, s.vector(), b)

    u, p = split(s)
    J = assemble(u**2*dx)
    Jhat = ReducedFunctional(J, Control(m))

    h = Constant(0.1)
    assert taylor_test(Jhat, m, h) > 1.9


def test_expression_function():
    mesh = UnitSquareMesh(3, 3)
    V_h = VectorElement("CG", mesh.ufl_cell(), 2)
    Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V_h * Q_h)

    M = FunctionSpace(mesh, "CG", 2)
    m = Function(M)
    m.vector()[:] = 2.0
    expr = Expression(("m*m*(x[1]-0.5)*(x[1]-0.5)", "0.3*m*m"), m=m, degree=1)
    expr.user_defined_derivatives = {m: Expression(("2*m*(x[1]-0.5)*(x[1]-0.5)", "0.6*m"), m=m, degree=1)}

    noslip = DirichletBC(W.sub(0), (0, 0),
                         "on_boundary && (x[1] >= 0.9 || x[1] < 0.1)")
    inflow = DirichletBC(W.sub(0), expr, "on_boundary && x[0] <= 0.1 && x[1] < 0.9 && x[1] > 0.1")

    bcs = [inflow, noslip]
    s = Function(W, name="State")
    v, q = TestFunctions(W)
    x = TrialFunction(W)
    u, p = split(x)

    nu = Constant(1)
    a = (nu * inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         - inner(q, div(u)) * dx
         )
    L = inner(Constant((0.0, 0.0)), v) * dx

    A, b = assemble_system(a, L, bcs)
    solve(A, s.vector(), b)

    u, p = split(s)
    J = assemble(u**2*dx)
    Jhat = ReducedFunctional(J, Control(m))

    h = Function(M)
    h.vector()[:] = rand(M.dim())
    assert taylor_test(Jhat, m, h) > 1.9


def test_expression_hessian_constant():
    mesh = UnitSquareMesh(3, 3)
    V_h = VectorElement("CG", mesh.ufl_cell(), 2)
    Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V_h * Q_h)

    m = Constant(2.0)
    expr = Expression(("m*m*m", "m*m*x[1]"), m=m, degree=1)
    deriv = Expression(("3*m*m", "2*m*x[1]"), m=m, tlm_input=1.0, degree=1)
    deriv.user_defined_derivatives = {m: Expression(("6*m", "2*x[1]"), m=m, degree=1)}
    expr.user_defined_derivatives = {m: deriv}

    noslip = DirichletBC(W.sub(0), (0, 0),
                         "on_boundary && (x[1] >= 0.9 || x[1] < 0.1)")
    inflow = DirichletBC(W.sub(0), expr, "on_boundary && x[0] <= 0.1 && x[1] < 0.9 && x[1] > 0.1")

    bcs = [inflow, noslip]
    s = Function(W, name="State")
    v, q = TestFunctions(W)
    x = TrialFunction(W)
    u, p = split(x)

    nu = Constant(1)
    a = (nu * inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         - inner(q, div(u)) * dx
         )
    L = inner(Constant((0.0, 0.0)), v) * dx

    A, b = assemble_system(a, L, bcs)
    solve(A, s.vector(), b)

    u, p = split(s)
    J = assemble(u**2*dx)
    Jhat = ReducedFunctional(J, Control(m))

    h = Constant(0.1)
    dJdm = Jhat.derivative()._ad_dot(h)
    Hm = Jhat.hessian(h)._ad_dot(h)
    assert taylor_test(Jhat, m, h, dJdm=dJdm, Hm=Hm) > 2.9


def test_expression_hessian_function():
    mesh = UnitSquareMesh(3, 3)
    V_h = VectorElement("CG", mesh.ufl_cell(), 2)
    Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V_h * Q_h)

    M = FunctionSpace(mesh, "CG", 2)
    m = Function(M)
    m.vector()[:] = 2.0
    expr = Expression(("m*m*m", "m*m"), m=m, degree=1)
    deriv = Expression(("3*m*m", "2*m"), m=m, tlm_input=1.0, degree=1)
    deriv.user_defined_derivatives = {m: Expression(("6*m", "2"), m=m, degree=1)}
    expr.user_defined_derivatives = {m: deriv}


    noslip = DirichletBC(W.sub(0), (0, 0),
                         "on_boundary && (x[1] >= 0.9 || x[1] < 0.1)")
    inflow = DirichletBC(W.sub(0), expr, "on_boundary && x[0] <= 0.1 && x[1] < 0.9 && x[1] > 0.1")

    bcs = [inflow, noslip]
    s = Function(W, name="State")
    v, q = TestFunctions(W)
    x = TrialFunction(W)
    u, p = split(x)

    nu = Constant(1)
    a = (nu * inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         - inner(q, div(u)) * dx
         )
    L = inner(Constant((0.0, 0.0)), v) * dx

    A, b = assemble_system(a, L, bcs)
    solve(A, s.vector(), b)

    u, p = split(s)
    J = assemble(u**2*dx)
    Jhat = ReducedFunctional(J, Control(m))

    h = Function(M)
    h.vector()[:] = rand(M.dim())
    dJdm = Jhat.derivative()._ad_dot(h)
    Hm = Jhat.hessian(h)._ad_dot(h)
    assert taylor_test(Jhat, m, h, dJdm=dJdm, Hm=Hm) > 2.9
