import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.__future__ import *
from firedrake.adjoint import *

from numpy.random import rand
from numpy.testing import assert_approx_equal, assert_allclose


def test_assign_linear_combination():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    x, = SpatialCoordinate(mesh)
    f = assemble(interpolate(x, V))
    g = assemble(interpolate(sin(x), V))
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
    f = assemble(interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V))
    g = assemble(interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V))
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
    f = assemble(interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V))
    g = assemble(interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V))
    u = Function(V)

    u.assign(f - 0.5*g)

    J = assemble(inner(f, g)*u**2*dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = 1
    f.block_variable.tlm_value = h

    tape = get_working_tape()
    tape.evaluate_tlm()

    assert J.block_variable.tlm_value is not None
    assert taylor_test(rf, f, h, dJdm=J.block_variable.tlm_value) > 1.9


def test_assign_tlm_with_constant():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(x[0], V))
    g = assemble(interpolate(sin(x[0]), V))
    c = Function(R, val=5.0)

    u = Function(V)
    u.interpolate(c * f**2)

    c.block_variable.tlm_value = Function(R, val=0.3)
    tape = get_working_tape()
    tape.evaluate_tlm()
    assert_allclose(u.block_variable.tlm_value.dat.data, 0.3 * f.dat.data ** 2)

    tape.reset_tlm_values()
    c.block_variable.tlm_value = Function(R, val=0.4)
    f.block_variable.tlm_value = g
    tape.evaluate_tlm()
    assert_allclose(u.block_variable.tlm_value.dat.data, 0.4 * f.dat.data ** 2 + 10. * f.dat.data * g.dat.data)


def test_assign_hessian():
    mesh = UnitSquareMesh(10, 10)
    element = VectorElement("CG", mesh.ufl_cell(), degree=1, dim=2)
    V = FunctionSpace(mesh, element)

    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(as_vector((x[0]*x[1], x[0]+x[1])), V))
    g = assemble(interpolate(as_vector((sin(x[1])+x[0], cos(x[0])*x[1])), V))
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
    f = assemble(interpolate(x[0], V))
    g = assemble(interpolate(sin(x[0]), V))
    u = Function(V)

    u.interpolate(f*g)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, f, h) > 1.9


def test_assign_with_constant():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    x = SpatialCoordinate(mesh)
    f = assemble(interpolate(x[0], V))
    c = Function(R, val=3.0)
    d = Function(R, val=2.0)
    u = Function(V)

    u.assign(c*f+d**3)

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
    f = assemble(interpolate(x[0], V))
    g = assemble(interpolate(sin(x[0]), V))
    control = Control(g)

    test = TestFunction(V)
    trial = TrialFunction(V)
    a = inner(grad(trial), grad(test))*dx
    L = inner(g, test)*dx

    bc = DirichletBC(V, g, "on_boundary")
    sol = Function(V)
    solve(a == L, sol, bc)

    u = Function(V)

    u.interpolate(f*sol*g)

    J = assemble(u ** 2 * dx)
    rf = ReducedFunctional(J, control)

    g = Function(V)
    g.vector()[:] = rand(V.dim())

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(rf, g, h) > 1.9


def test_assign_constant_scale():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)
    f = Function(V)
    c = Function(R, val=2.0)
    x, y = SpatialCoordinate(mesh)
    g = assemble(interpolate(as_vector([sin(y)+x, cos(x)*y]), V))

    f.assign(c * g)

    J = assemble(inner(f, f) ** 2  * dx)

    rf = ReducedFunctional(J, Control(c))
    r = taylor_to_dict(rf, c, Constant(0.1))

    assert min(r["R0"]["Rate"]) > 0.9
    assert min(r["R1"]["Rate"]) > 1.9
    assert min(r["R2"]["Rate"]) > 2.9


def test_assign_cofunction():
    # Cofunction assign test using 1D heat equation.
    mesh = IntervalMesh(10, 0, 1)
    x, = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    ic = project(sin(pi * x), V)
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V)
    u_n = Function(V).assign(ic)
    bc = DirichletBC(V, 0.0, "on_boundary")
    F = (u - u_n) / Constant(0.001) * v * dx + Constant(0.005) * inner(grad(u_n), grad(v)) * dx
    vom_mesh = VertexOnlyMesh(mesh, [[0.5]])
    vom_space = FunctionSpace(vom_mesh, "DG", 0)
    point_source = assemble(Constant(1.0) * TestFunction(vom_space)*dx)
    f_delta = Cofunction(V.dual()).interpolate(point_source)
    f = Cofunction(V.dual())
    linear_problem = LinearVariationalProblem(lhs(F), rhs(F) + f, u_np1, bcs=bc)
    linear_solver = LinearVariationalSolver(linear_problem)
    for t in range(2):
        f.assign(assemble(exp(-t) * f_delta))
        linear_solver.solve()
        u_n.assign(u_np1)
    J_val = assemble(u_np1*u_np1*dx)
    J_hat = ReducedFunctional(J_val, Control(ic))
    assert taylor_test(J_hat, ic, Function(V).assign(0.1)) > 1.9


