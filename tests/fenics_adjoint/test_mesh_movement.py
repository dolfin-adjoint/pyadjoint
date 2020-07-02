from dolfin import *
from dolfin_adjoint import *
from numpy import isclose, allclose
from numpy.linalg import norm as npnorm

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

def test_boundary_mesh_movement():
    mesh = UnitSquareMesh(25,25)
    b_mesh = BoundaryMesh(mesh, "exterior")
    S_b = VectorFunctionSpace(b_mesh, "CG", 1)
    movement = interpolate(Expression(("x[0]", "x[1]"), degree=1), S_b)
    v_volume = transfer_from_boundary(movement, mesh)
    v_reverse = transfer_to_boundary(v_volume, b_mesh)
    assert(allclose(assemble(inner(movement,movement)*dx),
                    assemble(inner(v_reverse, v_reverse)*dx)))
