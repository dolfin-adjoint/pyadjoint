from dolfin import *
from dolfin_adjoint import *
from numpy import isclose

def test_forced_movement():
    mesh = UnitSquareMesh(10,10)
    S = VectorFunctionSpace(mesh, "CG", 1)
    movement = interpolate(Expression(("0", "2*x[1]"), degree=1), S)
    v = interpolate(Expression(("x[0]","x[1]"),degree=1), S)
    ALE.move(mesh, v)
    s = Function(S)
    ALE.move(mesh, s)
    J = assemble(1*dx(domain=mesh))
    Jhat = ReducedFunctional(J, Control(s))
    assert(isclose(Jhat(s), 4))
    assert(isclose(Jhat(movement), 8))


def test_reset_movement():
    mesh = UnitSquareMesh(10,10)
    S = VectorFunctionSpace(mesh, "CG", 1)
    movement = interpolate(Expression(("x[0]", "0"), degree=1), S)
    s = Function(S)
    v = interpolate(Expression(("x[0]","x[1]"),degree=1), S)
    ALE.move(mesh, v)
    ALE.move(mesh, s, reset_mesh=True)
    J = assemble(1*dx(domain=mesh))
    Jhat = ReducedFunctional(J, Control(s))
    assert(isclose(Jhat(s), 1))
    assert(isclose(Jhat(movement), 2))
