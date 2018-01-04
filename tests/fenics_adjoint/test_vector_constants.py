import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *


def test_simple_assemble():
    mesh = IntervalMesh(10, 0, 1)

    c = Constant((3.0, 4.0))
    J = assemble(c**2*dx(domain=mesh))

    Jhat = ReducedFunctional(J, Control(c))
    deriv = Jhat.derivative()

    assert(taylor_test(Jhat, c, Constant((0.3, 0.5))) > 1.9)


def test_simple_solve():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    c = Constant((3.0, 4.0, 7.0))

    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)

    a = u*v*dx
    L = inner(c, c)*v*dx

    solve(a == L, sol)

    J = assemble(sol**2*dx)
    Jhat = ReducedFunctional(J, Control(c))

    assert(taylor_test(Jhat, c, Constant((1, 1, 1))) > 1.9)


def test_solve_assemble():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    c = Constant((3.0, 4.0, 7.0))

    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(c, c)*v*dx

    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(a == L, sol, bc)

    J = assemble(c**2*sol**2*dx)
    Jhat = ReducedFunctional(J, Control(c))

    assert(taylor_test(Jhat, c, Constant((1, 1, 1))) > 1.9)


def test_dirichlet_bc():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V_elem = VectorElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, V_elem)

    c = Constant((1.0, 2.0))

    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)
    f = Function(V)
    f.interpolate(Constant((0.5, 3.0)))

    a = inner(grad(u), grad(v))*dx
    L = inner(c, v)*dx

    bc = DirichletBC(V, c, "on_boundary")
    solve(a == L, sol, bc)

    J = assemble(sol**2*dx)
    Jhat = ReducedFunctional(J, Control(c))

    assert(taylor_test(Jhat, c, Constant((1, 1))) > 1.9)


def test_simple_assemble_hessian():
    tape = Tape()
    set_working_tape(tape)
    mesh = IntervalMesh(10, 0, 1)

    c = Constant((3.0, 4.0))
    J = assemble(c**2*c**2*dx(domain=mesh))
    control = Control(c)
    Jhat = ReducedFunctional(J, control)
    h = Constant((1.0, 1.0))
    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))

    assert(taylor_test(Jhat, c, h, dJdm=dJdm, Hm=Hm) > 2.9)


def test_solve_assemble_hessian():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "CG", 1)

    c = Constant((3.0, 4.0, 7.0))

    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(c, c)*v*dx

    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(a == L, sol, bc)

    J = assemble(c**2*sol**2*dx)
    control = Control(c)
    Jhat = ReducedFunctional(J, control)
    h = Constant((1, 1, 1))
    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))

    assert(taylor_test(Jhat, c, h, dJdm=dJdm, Hm=Hm) > 2.9)


def test_dirichlet_bc_hessian():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V_elem = VectorElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, V_elem)

    c = Constant((1.0, 2.0))

    u = TrialFunction(V)
    v = TestFunction(V)
    sol = Function(V)
    f = Function(V)
    f.interpolate(Constant((0.5, 3.0)))

    a = inner(grad(u), grad(v))*dx
    L = inner(c, v)*dx

    bc = DirichletBC(V, c, "on_boundary")
    solve(a == L, sol, bc)

    J = assemble(c**2*sol**2*dx)
    control = Control(c)
    Jhat = ReducedFunctional(J, control)
    h = Constant((1,1))
    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))

    assert(taylor_test(Jhat, c, h, dJdm=dJdm, Hm=Hm) > 2.9)



