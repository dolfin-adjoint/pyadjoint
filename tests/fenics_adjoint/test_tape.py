import pytest

from fenics import *
from fenics_adjoint import *

@pytest.mark.parametrize("outname,dot", [("tape", False), ("tape.dot", True)])
def test_tape_visualisation(outname, dot):
    if dot:
        pytest.importorskip("networkx")
        pytest.importorskip("pygraphviz")
    else:
        pytest.importorskip("tensorflow")
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 1

    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    a = f*inner(grad(u), grad(v))*dx
    L = f*v*dx
    solve(a == L, u_, bc)
    project(u_, V)

    tape = get_working_tape()
    tape.visualise()

@pytest.mark.parametrize("outname,dot", [("tape", False), ("tape.dot", True)])
def test_tape_time(outname, dot):
    if dot:
        pytest.importorskip("networkx")
        pytest.importorskip("pygraphviz")

    else:
        pytest.importorskip("tensorflow")
    set_working_tape(Tape())

    mesh = IntervalMesh(10, 0, 1)

    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    def left(x, on_boundary):
        return near(x[0],0)

    def right(x, on_boundary):
        return near(x[0],1)

    bc_left = DirichletBC(V, 1, left)
    bc_right = DirichletBC(V, 2, right)
    bc = [bc_left, bc_right]

    T = 1.0
    dt = 0.1
    f = Constant(1)

    u_1 = Function(V)
    u_1.vector()[:] = 1

    a = u_1*u*v*dx + dt*f*inner(grad(u),grad(v))*dx
    L = u_1*v*dx

    t = dt
    while t <= T:
        solve(a == L, u_, bc)
        u_1.assign(u_)
        t += dt

    assemble(u_1**2*dx)
    tape = get_working_tape()
    tape.visualise(outname)

@pytest.mark.parametrize("outname,dot", [("tape", False), ("tape.dot", True)])
def test_tape_time_visualisation(outname, dot):
    if dot:
        pytest.importorskip("networkx")
        pytest.importorskip("pygraphviz")

    else:
        pytest.importorskip("tensorflow")
    set_working_tape(Tape())

    mesh = IntervalMesh(10, 0, 1)

    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    def left(x, on_boundary):
        return near(x[0],0)

    def right(x, on_boundary):
        return near(x[0],1)

    bc_left = DirichletBC(V, 1, left)
    bc_right = DirichletBC(V, 2, right)
    bc = [bc_left, bc_right]

    T = 1.0
    T = 0.5
    dt = 0.1
    f = project(Constant(1, name="Source"), V)

    u_1 = Function(V)
    u_1.vector()[:] = 1

    a = u_1*u*v*dx + dt*f*inner(grad(u),grad(v))*dx
    L = u_1*v*dx

    tape = get_working_tape()

    t = dt
    while t <= T:
        with tape.name_scope("Timestep"):
            solve(a == L, u_, bc)
            u_1.assign(u_)
            t += dt

    assemble(u_1**2*dx)
    tape.visualise()

@pytest.mark.parametrize("outname,dot", [("tape", False), ("tape.dot", True)])
def test_visualise_negative_float(outname, dot):
    if dot:
        pytest.importorskip("networkx")
        pytest.importorskip("pygraphviz")

    else:
        pytest.importorskip("tensorflow")
    set_working_tape(Tape())
    a = AdjFloat(-1.0)
    b = AdjFloat(2.0)
    c = a + b

    tape = get_working_tape()
    tape.visualise()
