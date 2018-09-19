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


def test_dirichlet_bc_on_subspace():
    tape = Tape()
    set_working_tape(tape)

    mesh = UnitSquareMesh(10, 10)
    V_h = VectorElement("CG", mesh.ufl_cell(), 2)
    Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)

    W = FunctionSpace(mesh, V_h * Q_h)
    V = FunctionSpace(mesh, V_h)

    v, q = TestFunctions(W)
    x = TrialFunction(W)
    u, p = split(x)
    s = Function(W, name="State")

    c = Constant((0.0, 0.0))

    nu = Constant(1)

    a = (nu * inner(grad(u), grad(v)) * dx
         - inner(p, div(v)) * dx
         - inner(q, div(u)) * dx
         )
    f = Function(V)
    L = inner(f, v) * dx

    u_inflow = Expression(("x[1]*(10-x[1])/25", "0"), degree=2)
    noslip = DirichletBC(W.sub(0), c,
                         "on_boundary && (x[1] >= 0.9 || x[1] < 0.1)")
    inflow = DirichletBC(W.sub(0), u_inflow, "on_boundary && x[0] <= 0.1")
    bcs = [inflow, noslip]

    solve(a == L, s, bcs)

    J = assemble(inner(s, s)**2*dx)
    control = Control(c)
    Jhat = ReducedFunctional(J, control)
    h = Constant((0.3,0.7))
    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))

    assert(taylor_test(Jhat, c, h, dJdm=dJdm, Hm=Hm) > 2.9)



