import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

def test_tape_visualisation():
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
    tape.visualise(filename="graph.pdf")

@pytest.mark.skipif_module_is_missing("pygraphviz")
def test_tape_visualisation():
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
    tape.visualise(filename="graph.dot", dot=True)

def test_tape_time():
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
