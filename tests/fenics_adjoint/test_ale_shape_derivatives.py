import pytest
import numpy as np
from dolfin import *
from dolfin_adjoint import *
from dolfin_utils.test import fixture

@fixture
def mesh():
    return UnitSquareMesh(6, 6)

@fixture
def X(mesh):
    return SpatialCoordinate(mesh)


def test_sin_weak(mesh):
    f = Expression("sin(x[0])*x[1]*exp(x[0]/(x[1]+0.1))",
                   degree=3,domain=mesh)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)

    J = f * dx(domain=mesh)
    Jhat = ReducedFunctional(assemble(J), Control(s))
    tape = get_working_tape()
    #tape.visualise()
    computed = Jhat.derivative().vector().get_local()

    V = TestFunction(S)
    dJV = div(V)*f*dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)    

def test_sin_weak_spatial(mesh, X):
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S)
    ALE.move(mesh, s)

    J = sin(X[0]) * dx
    Jhat = ReducedFunctional(assemble(J), Control(s))
    computed = Jhat.derivative().vector().get_local()

    V = TestFunction(S)
    dJV = div(V)*sin(X[0])*dx + V[0]*cos(X[0])*dx
    actual = assemble(dJV).get_local()
    assert np.allclose(computed, actual, rtol=1e-14)



def test_tlm_assemble(mesh):
    tape = get_working_tape()
    tape.clear_tape()
    set_working_tape(tape)
    S =  VectorFunctionSpace(mesh, "CG", 1)
    h = Function(S)
    h.interpolate(Expression(("A*cos(x[1])", "A*x[1]"),degree=2,A=10))
    f = Function(S)
    f.interpolate(Expression(("A*sin(x[1])", "A*cos(x[1])"),degree=2,A=10))
    X = SpatialCoordinate(mesh)
    s = Function(S,name="deform")
    ALE.move(mesh, s)

    J = assemble(sin(X[1])* dx(domain=mesh))

    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    assert(r0 >0.95)
    Jhat(s)
    # Tangent linear model
    s.tlm_value = h
    tape = get_working_tape()
    tape.evaluate_tlm()
    r1_tlm = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert(r1_tlm > 1.9)
    Jhat(s)
    r1 = taylor_test(Jhat, s, h)
    assert(np.isclose(r1,r1_tlm, rtol=1e-14))


def test_shape_hessian(mesh):
    tape = get_working_tape()
    tape.clear_tape()
    set_working_tape(tape)
    X = SpatialCoordinate(mesh)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    
    ALE.move(mesh, s)
    J = assemble(sin(X[1])* dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    f = Function(S, name="W")
    f.interpolate(Expression(("A*sin(x[1])", "A*cos(x[1])"),degree=2,A=10))
    h = Function(S,name="V")
    h.interpolate(Expression(("A*cos(x[1])", "A*x[1]"),degree=2,A=10))
    

    # Second order taylor
    dJdm = Jhat.derivative().vector().inner(h.vector())
    Hm = compute_hessian(J, c, h).vector().inner(h.vector())
    r2 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert(r2 > 2.9)
    Jhat(s)
    dJdmm_exact = derivative(derivative(sin(X[1])* dx(domain=mesh),X,h), X, h)
    assert(np.isclose(assemble(dJdmm_exact), Hm))


def test_PDE_hessian(mesh):
    tape = get_working_tape()
    tape = Tape()
    set_working_tape(tape)
    X = SpatialCoordinate(mesh)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    ALE.move(mesh, s)
    f = X[0]*X[1]
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    l = f*v*dx
    bc = DirichletBC(V, Constant(1), "on_boundary")
    u = Function(V)
    solve(a==l, u, bcs=bc)

    J = assemble(u*dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    f = Function(S, name="W")
    f.interpolate(Expression(("A*sin(x[1])", "A*cos(x[1])"),degree=2,A=10))
    h = Function(S,name="V")
    h.interpolate(Expression(("A*cos(x[1])", "A*x[1]"),degree=2,A=10))

    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    Jhat(s)
    assert(r0>0.95)
    # First order taylor
    s.tlm_value = h
    tape = get_working_tape()
    tape.evaluate_tlm()
    r1 = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert(r1>1.95)
    Jhat(s)

    # Second order taylor
    dJdm = Jhat.derivative().vector().inner(h.vector())
    Hm = compute_hessian(J, c, h).vector().inner(h.vector())
    r2 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert(r2>2.95)
