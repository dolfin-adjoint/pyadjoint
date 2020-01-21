from fenics import *
from fenics_adjoint import *


def test_preserved_ufl_shape():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Constant(1)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v))*dx - f**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c**2*u*dx)
    Jhat = ReducedFunctional(J, Control(c))
    dJdc = Jhat.derivative()

    assert dJdc.ufl_shape == c.ufl_shape


def test_reinitialization():
    mesh = UnitIntervalMesh(10)

    c = AdjFloat(2.0)
    J = assemble(Constant(c) ** 2 * dx(domain=mesh))
    Jhat = ReducedFunctional(J, Control(c))
    assert Jhat(2.0) == J
    assert Jhat.derivative() == 4.
