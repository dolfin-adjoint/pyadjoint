import pytest
import numpy as np
from dolfin import *
from dolfin_adjoint import *

def test_femorph_weak_shape_derivative():
    # Setup AD problem
    mesh = Mesh(UnitSquareMesh(6, 6), WeakForm=True)
    n = FacetNormal(mesh)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    V = TestFunction(S)

    # Problem specific functions
    f = Expression("x[0]*x[1]*sin(x[0])*cos(x[1])", degree=3, domain=mesh)
    u = Expression("x[0]*x[0]+x[1]*x[0]+x[1]*x[1]+sin(x[0])+cos(x[0])",
                   degree=2, domain=mesh)
    

    # Standard Expression
    J = f * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    dJV = f*div(V) * dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

    
    # Standard Expression squared
    J = u * u * dx(domain=mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    actual = assemble(u * u * div(V) * dx(domain=mesh)).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

    # Boundary integral
    J = f * ds(domain=mesh)
    from femorph import VolumeNormal
    n = VolumeNormal(mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    dJV = f * (div(V) - inner(dot(grad(V),n), n))*ds
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

    
@pytest.mark.xfail(strict=True,reason="Weak derivative not working properly")
def test_femorph_weak_mode():
    # Setup AD problem
    mesh = Mesh(UnitSquareMesh(6, 6), WeakForm=True)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    V = TestFunction(S)

    # Note: Florians implementation has
    # dJV = -2*inner(dot(grad(V), grad(u)), grad(u)) * dx + inner(grad(u), grad(u)) * div(V) * dx, which means that he also includes the second term in eq 7 stephans paper
    u = Expression("x[0]*x[0]+x[1]*x[0]+x[1]*x[1]+sin(x[0])+cos(x[0])",
                   degree=2, domain=mesh)
    J = inner(grad(u), grad(u)) * dx(domain=mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    dJV = inner(grad(u), grad(u)) * div(V) * dx - 2*inner(dot(grad(V), grad(u)), grad(u)) * dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

def test_femorph_mode_1():
    # Setup AD problem
    mesh = Mesh(UnitSquareMesh(6, 6), WeakForm=1)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    V = TestFunction(S)

    u = Expression("x[0]*x[0]+x[1]*x[0]+x[1]*x[1]+sin(x[0])+cos(x[0])",
                   degree=2, domain=mesh)
    J = inner(grad(u), grad(u)) * dx(domain=mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    dJV = div(inner(grad(u),grad(u))*V)*dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    
    
    
def test_femorph_strong_shape_derivative():
    mesh = Mesh(UnitSquareMesh(6, 6))
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    
    u = Expression("x[0]*x[0]+x[1]*x[0]+x[1]*x[1]+sin(x[0])+cos(x[0])",
                   degree=2)
    V = TestFunction(VectorFunctionSpace(mesh, "Lagrange", 1))
    J = u * u * dx(domain=mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))
    
    computed = Jhat.derivative().vector().get_local()
    from femorph import VolumeNormal
    # Note: Does not work with n=FacetNormal(mesh)
    n = VolumeNormal(mesh)
    actual = assemble(inner(n, V)*u*u*ds(domain=mesh)).get_local()
    for i in range(len(actual)):
        print(computed[i], actual[i])
    assert np.allclose(computed, actual, rtol=1e-14)

    
# def test_mixed_derivatives():
# mesh = Mesh(UnitSquareMesh(6, 6))
# S = VectorFunctionSpace(mesh, "CG", 1)
# s = Function(S)
# ALE.move(mesh, s)
# V = FunctionSpace(mesh, "CG", 1)
# u = Expression("x[0]*x[0]+x[1]*x[0]+x[1]*x[1]+sin(x[0])+cos(x[0])", degree=3, domain=mesh)
# v = TrialFunction(V)
# dX = TestFunction(S)
# dX_ = TrialFunction(S)

# J = u * u * dx
# Jhat = ReducedFunctional(J, Control(s))

# from IPython import embed; embed()
    # computed1 = assemble(derivative(derivative(J, X, dX), u)).array()
    # computed2 = assemble(derivative(derivative(J, u), X, dX_)).array()
    # actual = assemble(2 * u * v * div(dX) * dx).array()
    # assert np.allclose(computed1, actual, rtol=1e-14)    
    # assert np.allclose(computed2.T, actual, rtol=1e-14)    

    # J = inner(grad(u), grad(u)) * dx
    # computed1 = assemble(derivative(derivative(J, X, dX), u)).array()
    # computed2 = assemble(derivative(derivative(J, u), X)).array()
    # actual = assemble(2*inner(grad(u), grad(v)) * div(dX) * dx
    #                   - 2*inner(dot(grad(dX), grad(u)), grad(v)) * dx 
    #                   - 2*inner(grad(u), dot(grad(dX), grad(v))) * dx).array()
    # assert np.allclose(computed1, actual, rtol=1e-14)    
    # assert np.allclose(computed2.T, actual, rtol=1e-14)    
