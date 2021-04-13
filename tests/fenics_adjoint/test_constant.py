from fenics import *
from fenics_adjoint import *

from numpy.random import randn


def test_preserved_ufl_shape():
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
    dJdc = Jhat.derivative()

    assert dJdc.ufl_shape == c.ufl_shape


def test_assign_float():
    af = AdjFloat(0.1)
    uc = Constant(0.0)
    uc.assign(af)
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = project(uc, V)
    J = assemble(u ** 4 * dx)
    rf = ReducedFunctional(J, Control(af))

    h = AdjFloat(0.1)
    rates = taylor_to_dict(rf, af, h)

    assert min(rates["R0"]["Rate"]) > 0.95
    assert min(rates["R1"]["Rate"]) > 1.95
    assert min(rates["R2"]["Rate"]) > 2.95


def test_annotate_init():
    mesh = UnitSquareMesh(1, 1)
    c = Constant(1.0)
    A = assemble(c * dx(domain=mesh))
    A_const = Constant(A)
    J = assemble(A_const ** 4 * dx(domain=mesh))
    rf = ReducedFunctional(J, Control(c))

    h = Constant(0.1)
    rates = taylor_to_dict(rf, c, h)

    assert min(rates["R0"]["Rate"]) > 0.95
    assert min(rates["R1"]["Rate"]) > 1.95
    assert min(rates["R2"]["Rate"]) > 2.95


def test_assign_matrix_constant():
    domain = UnitSquareMesh(1, 1)
    c = Constant([[1., 2.], [3., 4.]])
    c2 = Constant([[0., 0.], [0., 0.]])

    c2.assign(c)

    J = assemble(inner(c2, c2) ** 2 * dx(domain=domain))
    Jhat = ReducedFunctional(J, Control(c))

    h = Constant(randn(2, 2))
    rates = taylor_to_dict(Jhat, c, h)
    assert min(rates["R0"]["Rate"]) > 0.95
    assert min(rates["R1"]["Rate"]) > 1.95
    assert min(rates["R2"]["Rate"]) > 2.95


def test_assign_vector_constant():
    domain = UnitSquareMesh(1, 1)
    c = Constant([1, 1.])
    c2 = Constant([0, 0])

    c2.assign(c)

    J = assemble(inner(c2, c2) ** 2 * dx(domain=domain))
    Jhat = ReducedFunctional(J, Control(c))

    h = Constant([0.1, 0.1])
    rates = taylor_to_dict(Jhat, c, h)
    assert min(rates["R0"]["Rate"]) > 0.95
    assert min(rates["R1"]["Rate"]) > 1.95
    assert min(rates["R2"]["Rate"]) > 2.95


def test_annotate_list():
    domain = UnitSquareMesh(1, 1)
    m = AdjFloat(1.0)
    c = Constant([m, 2.])

    J = assemble(inner(c, c) ** 2 * dx(domain=domain))
    Jhat = ReducedFunctional(J, Control(m))

    h = AdjFloat(0.1)
    rates = taylor_to_dict(Jhat, m, h)
    assert min(rates["R0"]["Rate"]) > 0.95
    assert min(rates["R1"]["Rate"]) > 1.95
    assert min(rates["R2"]["Rate"]) > 2.95

