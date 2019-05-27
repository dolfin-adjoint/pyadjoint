from dolfin import *
from dolfin_adjoint import *
from numpy.random import rand


def test_isolated_function_call_scalar():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.vector()[:] = rand(V.dim())
    J = f(rand(2))**2

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(ReducedFunctional(J, Control(f)), f, h) > 1.9


def test_function_call_scalar():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    X, Y = SpatialCoordinate(mesh)
    f = project(sin(2*pi*X*Y), V)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx
    bc = DirichletBC(V, 1, "on_boundary")
    U = Function(V)
    solve(a == L, U, bc)
    J = U(rand(2)) * U(rand(2))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(ReducedFunctional(J, Control(f)), f, h) > 1.9


def test_isolated_function_call_vector():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.vector()[:] = rand(V.dim())
    J = f(rand(2))
    J = (J[0] + J[1])**2

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(ReducedFunctional(J, Control(f)), f, h) > 1.9


def test_function_call_vector():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 1)

    X, Y = SpatialCoordinate(mesh)
    f = project(as_vector((sin(2*pi*X*Y), Y)), V)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx
    bc = DirichletBC(V, (1, 2), "on_boundary")
    U = Function(V)
    solve(a == L, U, bc)
    J = U(rand(2))[1]**2

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(ReducedFunctional(J, Control(f)), f, h) > 1.9

