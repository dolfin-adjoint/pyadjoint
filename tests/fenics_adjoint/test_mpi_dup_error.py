import pytest
pytest.importorskip("fenics")
pytest.importorskip("mpi4py")

from fenics import *
from fenics_adjoint import *

import mpi4py
import gc


_comm = MPI.comm_world


def get_free_comms():
    """Compute the number of free MPI communicators
    I could not find a direct way of doing this, so I made this hack.
    """
    i = 0

    class temporary_communicators():
        def __init__(self):
            self.tmp_comms = []

        def __enter__(self):
            return self.tmp_comms

        def __exit__(self, *args):
            for tmp_comm in self.tmp_comms:
                tmp_comm.Free()

    with temporary_communicators() as tmp_comms:
        try:
            while i <= 100000:
                tmp_comms.append(_comm.Dup())
                i += 1
            raise RuntimeError("Could not count free MPI communicators.")
        except mpi4py.MPI.Exception:
            return i


def test_expression_adj_fs():
    """This test is to see if there is any MPI communicator leakage for expressions (due to function spaces).
    The bug that initiated this test was due to Expressions creating and storing function spaces
    in the derivative routine, which was duplicated for each time the expression was used in the forward code.
    """
    mesh = IntervalMesh(2, 0, 1)
    # Allocate functionspace and f to create the related MPI communicators.
    # This way it becomes easier to count the correct number of MPI comms used.
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)

    gc.collect()
    original = get_free_comms()
    c = Constant(1.0)
    expr = Expression("c", c=c, degree=1)

    J = 0.
    for i in range(100):
        J += assemble(expr*dx(domain=mesh))

    Jhat = ReducedFunctional(J, Control(c))
    Jhat.derivative()
    gc.collect()
    # Ideally this would be original == get_free_comms(),
    # but we still (currently) need to create and persist a function space for each expression.
    assert original - get_free_comms() <= 1


def test_dirichletbc_subspace():
    """This test checks if there is any MPI communicator leakage for DirichletBC defined on subspaces.
    This was a bug because `FunctionSpace.collapse()` creates a new function space, which meant that
    each time a DirichletBC is applied to a PDE and that PDE is solved, a new (collapsed) function space is created.
    """
    mesh = UnitSquareMesh(3, 3)
    V = VectorFunctionSpace(mesh, "CG", 1)
    sol = Function(V)

    gc.collect()
    original = get_free_comms()

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = inner(Constant((0, 0)), v)*dx

    bc = DirichletBC(V.sub(0), Constant(1), "on_boundary")
    A, b = assemble_system(a, L, bc)

    for i in range(100):
        solve(A, sol.vector(), b)

    gc.collect()
    assert original - get_free_comms() <= 1
