from fenics import *
from fenics_adjoint import *
from numpy.random import rand
import pytest


def test_simple():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    f.vector()[:] = 1
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A = assemble(a)
    b = assemble(L)

    bc = DirichletBC(V, 1, "on_boundary")
    bc.apply(A, b)

    solver = PETScKrylovSolver()
    solver.set_operator(A)

    sol = Function(V)
    solver.solve(sol.vector(), b)

    J = assemble(inner(sol, sol)*dx)
    Jhat = ReducedFunctional(J, Control(f))

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(Jhat, f, h) > 1.9


def test_petsc_options():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    f.vector()[:] = 1
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A = assemble(a)
    b = assemble(L)

    bc = DirichletBC(V, 1, "on_boundary")
    bc.apply(A, b)

    PETScOptions.set("ksp_type", "gmres")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("pc_hypre_type", "boomeramg")
    PETScOptions.set("pc_hypre_boomeramg_max_iter", 1)
    PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.7)
    PETScOptions.set("pc_hypre_boomeramg_coarsen_type", "HMIS")
    PETScOptions.set("pc_hypre_boomeramg_agg_nl", 4)

    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.set_operator(A)

    sol = Function(V)
    solver.solve(sol.vector(), b)

    J = assemble(inner(sol, sol)*dx)
    Jhat = ReducedFunctional(J, Control(f))
    assert Jhat(f) == J

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    assert taylor_test(Jhat, f, h) > 1.7

