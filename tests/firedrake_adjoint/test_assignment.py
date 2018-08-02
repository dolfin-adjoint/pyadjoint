import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake_adjoint import *

from numpy.random import rand

# To access Expressions - they are disabled in firedrake_adjoint (even without annotate)
import firedrake as fd


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

    f = interpolate(fd.Expression(("x[0]*x[1]", "x[0]+x[1]"), degree=1), V)
    g = interpolate(fd.Expression(("sin(x[1])+x[0]", "cos(x[0])*x[1]"), degree=1), V)
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

    f = interpolate(fd.Expression(("x[0]*x[1]", "x[0]+x[1]"), degree=1), V)
    g = interpolate(fd.Expression(("sin(x[1])+x[0]", "cos(x[0])*x[1]"), degree=1), V)
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


def test_assign_hessian():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    f = interpolate(fd.Expression(("x[0]*x[1]", "x[0]+x[1]"), degree=1), V)
    g = interpolate(fd.Expression(("sin(x[1])+x[0]", "cos(x[0])*x[1]"), degree=1), V)
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

    f = interpolate(fd.Expression("x[0]", degree=1), V)
    g = interpolate(fd.Expression("sin(x[0])", degree=1), V)
    u = Function(V)

    u.assign(f*g)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, f, h) > 1.9


def test_assign_nonlin_changing():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    f = interpolate(fd.Expression("x[0]", degree=1), V)
    g = interpolate(fd.Expression("sin(x[0])", degree=1), V)
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
