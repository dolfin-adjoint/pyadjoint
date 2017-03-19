from fenics import *
from fenics_adjoint import *
import networkx as nx

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

    tape = get_working_tape()
    tape.visualise(filename="graph.pdf")