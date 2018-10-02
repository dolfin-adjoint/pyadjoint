import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def test_assign_linear_combination():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    f = interpolate(Expression("x[0]", degree=1), V)
    g = interpolate(Expression("sin(x[0])", degree=1), V)
    u = Function(V)

    u.assign(3*f + g)

    J = assemble(u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, f, h) > 1.9


def test_assign_vector_valued():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    f = interpolate(Expression(("x[0]*x[1]", "x[0]+x[1]"), degree=1), V)
    g = interpolate(Expression(("sin(x[1])+x[0]", "cos(x[0])*x[1]"), degree=1), V)
    u = Function(V)

    u.assign(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, f, h) > 1.9


def test_assign_tlm():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    f = interpolate(Expression(("x[0]*x[1]", "x[0]+x[1]"), degree=1), V)
    g = interpolate(Expression(("sin(x[1])+x[0]", "cos(x[0])*x[1]"), degree=1), V)
    u = Function(V)

    u.assign(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    f.tlm_value = h

    tape = get_working_tape()
    tape.evaluate_tlm()

    assert taylor_test(rf, f, h, dJdm=J.tlm_value) > 1.9


def test_assign_hessian():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    f = interpolate(Expression(("x[0]*x[1]", "x[0]+x[1]"), degree=1), V)
    g = interpolate(Expression(("sin(x[1])+x[0]", "cos(x[0])*x[1]"), degree=1), V)
    u = Function(V)

    u.assign(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    dJdm = rf.derivative()

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    Hm = rf.hessian(h)
    assert taylor_test(rf, f, h, dJdm=h._ad_dot(dJdm), Hm=h._ad_dot(Hm)) > 2.9


def test_assign_nonlincom_error():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    f = interpolate(Expression("x[0]", degree=1), V)
    g = interpolate(Expression("sin(x[0])", degree=1), V)
    u = Function(V)

    with pytest.raises(RuntimeError): u.assign(f*g)

