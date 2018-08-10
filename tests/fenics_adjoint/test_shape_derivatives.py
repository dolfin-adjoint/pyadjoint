import pytest
import numpy as np
from dolfin import *
from dolfin_adjoint import *
from dolfin_utils.test import fixture

@fixture
def WeakMesh():
    return UnitSquareMesh(6, 6)

@fixture
def X(WeakMesh):
    return SpatialCoordinate(WeakMesh)


def test_sin_weak(WeakMesh):
    f = Expression("sin(x[0])*x[1]*exp(x[0]/(x[1]+0.1))",
                   degree=3, domain=WeakMesh)
    S = VectorFunctionSpace(WeakMesh, "CG", 1)
    s = Function(S)
    ALE.move(WeakMesh, s)

    J = f * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()

    V = TestFunction(S)
    dJV = div(V)*f*dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

def test_sin_weak_spatial(WeakMesh, X):
    S = VectorFunctionSpace(WeakMesh, "CG", 1)
    s = Function(S)
    ALE.move(WeakMesh, s)

    J = sin(X[0]) * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()

    V = TestFunction(S)
    dJV = div(V)*sin(X[0])*dx + V[0]*cos(X[0])*dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

@fixture
def StrongMesh():
    return UnitSquareMesh(6, 6, hadamard_form=True)

@fixture
def n(StrongMesh):
    return FacetNormal(StrongMesh)

def test_strong_strong(StrongMesh, n):
    femorph = pytest.importorskip("femorph")
    S = VectorFunctionSpace(StrongMesh, "CG", 1)
    s = Function(S)
    ALE.move(StrongMesh, s)

    f = Expression("sin(x[0])*x[1]*exp(x[0]/(x[1]+0.10))", degree=3,
                   domain=StrongMesh)
    J = f * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()

    V = TestFunction(S)
    dJV = inner(V, n)*f*ds
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)
