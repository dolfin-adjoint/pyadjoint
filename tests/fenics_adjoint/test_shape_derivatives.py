import pytest
import numpy as np
from dolfin import *
from dolfin_adjoint import *

def test_femorph_weak_shape_derivative():
    mesh = Mesh(UnitSquareMesh(6, 6), WeakForm=True)
    n = FacetNormal(mesh)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)

    u = Expression("x[0]*x[0]+x[1]*x[0]+x[1]*x[1]+sin(x[0])+cos(x[0])",
                   degree=2)
    dX = TestFunction(VectorFunctionSpace(mesh, "Lagrange", 1))
    J = u * u * dx(domain=mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))

    computed = Jhat.derivative().vector().get_local()
    actual = assemble(u * u * div(dX) * dx(domain=mesh)).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

def test_femorph_strong_shape_derivative():
    mesh = Mesh(UnitSquareMesh(6, 6))
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    
    u = Expression("x[0]*x[0]+x[1]*x[0]+x[1]*x[1]+sin(x[0])+cos(x[0])",
                   degree=2)
    dX = TestFunction(VectorFunctionSpace(mesh, "Lagrange", 1))
    J = u * u * dx(domain=mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))
    
    computed = Jhat.derivative().vector().get_local()
    from femorph import VolumeNormal
    # Note: Does not work with n=FacetNormal(mesh)
    n = VolumeNormal(mesh)
    actual = assemble(inner(n, dX)*u*u*ds(domain=mesh)).get_local()
    for i in range(len(actual)):
        print(computed[i], actual[i])
    assert np.allclose(computed, actual, rtol=1e-14)
    
