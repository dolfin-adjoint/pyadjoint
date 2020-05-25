import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake_adjoint import *

from numpy.random import rand
from numpy.testing import assert_approx_equal, assert_allclose


def test_assign_linear_combination():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x, = SpatialCoordinate(mesh)
    f = interpolate(x, V)
    g = interpolate(sin(x), V)
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

    x = SpatialCoordinate(mesh)
    f = interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V)
    g = interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V)
    u = Function(V)

    u.assign(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = 1
    assert taylor_test(rf, f, h) > 1.9


def test_assign_tlm():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    x = SpatialCoordinate(mesh)
    f = interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V)
    g = interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V)
    u = Function(V)

    u.assign(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = 1
    f.tlm_value = h

    tape = get_working_tape()
    tape.evaluate_tlm()

    assert taylor_test(rf, f, h, dJdm=J.tlm_value) > 1.9


def test_assign_tlm_wit_constant():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    f = interpolate(x[0], V)
    g = interpolate(sin(x[0]), V)
    c = Constant(5.0)

    u = Function(V)
    u.assign(c * f ** 2)

    c.tlm_value = Constant(0.3)
    tape = get_working_tape()
    tape.evaluate_tlm()
    assert_allclose(u.block_variable.tlm_value.dat.data, 0.3 * f.dat.data ** 2)

    tape.reset_tlm_values()
    c.tlm_value = Constant(0.4)
    f.tlm_value = g
    tape.evaluate_tlm()
    assert_allclose(u.block_variable.tlm_value.dat.data, 0.4 * f.dat.data ** 2 + 10. * f.dat.data * g.dat.data)


def test_assign_hessian():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    x = SpatialCoordinate(mesh)
    f = interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V)
    g = interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V)
    u = Function(V)

    u.assign(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    dJdm = rf.derivative()

    h = Function(V)
    h.vector()[:] = 1.0
    Hm = rf.hessian(h)
    assert taylor_test(rf, f, h, dJdm=h._ad_dot(dJdm), Hm=h._ad_dot(Hm)) > 2.9


def test_assign_nonlincom():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    f = interpolate(x[0], V)
    g = interpolate(sin(x[0]), V)
    u = Function(V)

    u.assign(f*g)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, f, h) > 1.9


def test_assign_with_constant():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    f = interpolate(x[0], V)
    c = Constant(3.0)
    d = Constant(2.0)
    u = Function(V)

    u.assign(c*f+d**3)

    # J = c**2/3 + cd**3 + d**6
    J = assemble(u ** 2 * dx)

    rf = ReducedFunctional(J, Control(c))
    dJdc = rf.derivative()
    assert_approx_equal(float(dJdc), 10.)

    rf = ReducedFunctional(J, Control(d))
    dJdd = rf.derivative()
    assert_approx_equal(float(dJdd), 228.)

def test_assign_nonlin_changing():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x = SpatialCoordinate(mesh)
    f = interpolate(x[0], V)
    g = interpolate(sin(x[0]), V)
    control = Control(g)

    test = TestFunction(V)
    trial = TrialFunction(V)
    a = inner(grad(trial), grad(test))*dx
    L = inner(g, test)*dx

    bc = DirichletBC(V, g, "on_boundary")
    sol = Function(V)
    solve(a == L, sol, bc)

    u = Function(V)

    u.assign(f*sol*g)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, control)

    g = Function(V)
    g.vector()[:] = rand(V.dim())

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, g, h) > 1.9
