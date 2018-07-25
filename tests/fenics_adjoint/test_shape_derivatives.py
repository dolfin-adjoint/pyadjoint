import pytest
import numpy as np
from dolfin import *
from dolfin_adjoint import *

def test_x2_strong():
    mesh = UnitSquareMesh(6, 6, hadamard_form=True)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    V = TestFunction(S)
    f = Expression("x[0]*x[0]", degree=3, domain=mesh)
    J = f * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    dJV = inner(V, FacetNormal(mesh))*f*ds
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    


def test_x2_div():
    mesh = UnitSquareMesh(6, 6, hadamard_form=False)
    f = Expression("x[0]*x[0]", degree=3, domain=mesh)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    V = TestFunction(S)
    J = f * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    dJV = div(V*f)*dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

def test_sin_strong():
    mesh = UnitSquareMesh(6, 6, hadamard_form=True)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    V = TestFunction(S)
    f = Expression("sin(x[0])", degree=3, domain=mesh)
    df = Expression(("-cos(x[0])", "0"), degree=3, domain=mesh)
    J = f * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    dJV = inner(V, FacetNormal(mesh))*f*ds# - inner(df,V)*dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    
    tape.clear_tape()

# @pytest.mark.xfail(strict=True,reason="Femorph not computing f'(x)[V]")
def test_sin_div():
    mesh = UnitSquareMesh(6, 6, hadamard_form=False)
    f = Expression("sin(x[0])", degree=3, domain=mesh)
    df = Expression(("-cos(x[0])","0"), degree=3, domain=mesh)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    V = TestFunction(S)
    J = f * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()
    dJV = div(V*f)*dx# - inner(df, V)*dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    


    
def test_femorph_mode_div():
    # Setup AD problem
    mesh = UnitSquareMesh(6, 6, hadamard_form=False)
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
    mesh = UnitSquareMesh(6, 6, hadamard_form=True)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)
    
    u = Expression("x[0]*x[0]+x[1]*x[0]+x[1]*x[1]+sin(x[0])+cos(x[0])",
                   degree=2)
    V = TestFunction(VectorFunctionSpace(mesh, "Lagrange", 1))
    J = u * u * dx(domain=mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))
    
    computed = Jhat.derivative().vector().get_local()
    n = FacetNormal(mesh)
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
