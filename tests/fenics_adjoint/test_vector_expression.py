import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def test_simple_constant():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10, 10)
    V_element = VectorElement("CG", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, V_element)

    u = TrialFunction(V)
    v = TestFunction(V)

    c = Constant(1.0)
    expr = Expression(("c*c", "2*c"), c=c, degree=1)
    expr.user_defined_derivatives = {c: Expression(("2*c", "2"), c=c, degree=1, annotate=False)}

    a = inner(u,v)*dx
    L = inner(expr, v)*dx

    bc = DirichletBC(V, (0, 0), "on_boundary")
    u_ = Function(V)
    solve(a == L, u_, bc)

    J = assemble(inner(u_, u_)*dx)
    Jhat = ReducedFunctional(J, c)
    assert(taylor_test(Jhat, c, Constant(1)) > 1.9)


def test_simple_function():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10, 10)
    V_element = VectorElement("CG", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, V_element)

    u = TrialFunction(V)
    v = TestFunction(V)

    W = FunctionSpace(mesh, "CG", 3)
    c = project(Expression("sin(x[0])*sin(x[1])", degree=2, annotate=False), W, annotate=False)
    expr = Expression(("c*c", "2*c"), c=c, degree=3)
    expr.user_defined_derivatives = {c: Expression(("2*c", "2"), c=c, degree=3, annotate=False)}

    a = inner(u,v)*dx
    L = inner(expr, v)*dx

    bc = DirichletBC(V, (0, 0), "on_boundary")
    u_ = Function(V)
    solve(a == L, u_, bc)

    J = assemble(inner(u_, u_)*dx)
    Jhat = ReducedFunctional(J, c)

    h = Function(c.function_space())
    h.vector()[:] = rand(c.function_space().dim())
    assert(taylor_test(Jhat, c, h) > 1.9)


def test_assemble_expr():
    tape = Tape()
    set_working_tape(tape)
    mesh = IntervalMesh(10, 0, 1)

    V = FunctionSpace(mesh, "CG", 2)
    c = project(Expression("exp(x[0])", degree=3, annotate=False), V, annotate=False)
    expr = Expression(("sin(c*x[0])", "cos(c*x[0])"), c=c, degree=2)
    expr.user_defined_derivatives = {c: Expression(("x[0]*cos(c*x[0])", "-x[0]*sin(c*x[0])"), c=c, degree=2, annotate=False)}

    J = assemble(inner(expr, expr) * dx(domain=mesh))
    Jhat = ReducedFunctional(J, c)

    h = Function(c.function_space())
    h.vector()[:] = rand(c.function_space().dim())
    assert (taylor_test(Jhat, c, h) > 1.9)


def test_scalar_working():
    tape = Tape()
    set_working_tape(tape)
    mesh = UnitSquareMesh(10, 10)
    V_element = FiniteElement("CG", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, V_element)

    u = TrialFunction(V)
    v = TestFunction(V)

    W = FunctionSpace(mesh, "CG", 3)
    c = project(Expression("sin(x[0])*sin(x[1])", degree=2, annotate=False), W, annotate=False)
    expr = Expression("c*c", c=c, degree=3)
    expr.user_defined_derivatives = {c: Expression("2*c", c=c, degree=3, annotate=False)}

    a = inner(u,v)*dx
    L = inner(expr, v)*dx

    bc = DirichletBC(V, 0, "on_boundary")
    u_ = Function(V)
    solve(a == L, u_, bc)

    J = assemble(inner(u_, u_)*dx)
    Jhat = ReducedFunctional(J, c)

    h = Function(c.function_space())
    h.vector()[:] = rand(c.function_space().dim())
    assert(taylor_test(Jhat, c, h) > 1.9)
