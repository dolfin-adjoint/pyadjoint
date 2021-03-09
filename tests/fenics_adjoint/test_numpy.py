import pytest

pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand, seed
seed(21)


def test_function_mul_float():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Constant(1)
    f = Function(V)
    with stop_annotating():
        f.vector()[:] = 2

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v)) * dx - f ** 2 * v * dx
    solve(F == 0, u, bc)
    mul = AdjFloat(3)
    u.vector()[:] *= mul

    J = assemble(c ** 2 * u ** 4 * dx)
    Jhat = ReducedFunctional(J, [Control(f), Control(mul)])

    assert J == Jhat([f, mul])

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    hmul = AdjFloat(1.0)
    r = taylor_to_dict(Jhat, [f, mul], [h, hmul])

    assert min(r["R0"]["Rate"]) > 0.9
    assert min(r["R1"]["Rate"]) > 1.9
    assert min(r["R2"]["Rate"]) > 2.9


if __name__ == "__main__":
    test_function_mul_float()
