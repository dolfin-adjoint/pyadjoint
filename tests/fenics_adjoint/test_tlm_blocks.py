from dolfin import *
from dolfin_adjoint import *
import pytest

@pytest.mark.xfail(reason="Not implemented yet")
def test_div_block():
    mesh = UnitSquareMesh(10,10)
    S = VectorFunctionSpace(mesh, "CG", 1)
    X = SpatialCoordinate(mesh)
    s = Function(S)
    ALE.move(mesh, s)
    
    J = assemble(X[0]*dx(domain=mesh))/assemble(1*dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)
    
    h = Function(S,name="V")
    h.interpolate(Expression(("sin(x[1])", "10*x[1]*x[0]"),degree=2))
    
    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    Jhat(s)
    assert(r0>0.95)
    s.tlm_value = h
    tape.evaluate_tlm()
    r1 = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert(r1 > 1.95)
    Jhat(s)
    dJdm = Jhat.derivative().vector().inner(h.vector())
    Hm = compute_hessian(J, c, h).vector().inner(h.vector())
    r3 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert(r3 > 2.95)

@pytest.mark.xfail(reason="Not implemented yet")
def test_pow_block():
    mesh = UnitSquareMesh(10,10)
    S = VectorFunctionSpace(mesh, "CG", 1)
    X = SpatialCoordinate(mesh)
    s = Function(S)
    ALE.move(mesh, s)
    
    J = (assemble(X[0]*dx(domain=mesh))-0.3)**2
    c = Control(s)
    Jhat = ReducedFunctional(J, c)
    
    h = Function(S,name="V")
    h.interpolate(Expression(("sin(x[1])", "10*x[1]*x[0]"),degree=2))
    
    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    Jhat(s)
    assert(r0>0.95)
    s.tlm_value = h
    tape.evaluate_tlm()
    r1 = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert(r1 > 1.95)
    Jhat(s)
    dJdm = Jhat.derivative().vector().inner(h.vector())
    Hm = compute_hessian(J, c, h).vector().inner(h.vector())
    r3 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert(r3 > 2.95)

@pytest.mark.xfail(reason="Not implemented yet")
def test_neg_block():
    mesh = UnitSquareMesh(10,10)
    S = VectorFunctionSpace(mesh, "CG", 1)
    X = SpatialCoordinate(mesh)
    s = Function(S)
    ALE.move(mesh, s)
    
    J = -assemble(X[0]*dx(domain=mesh))-0.3
    c = Control(s)
    Jhat = ReducedFunctional(J, c)
    
    h = Function(S,name="V")
    h.interpolate(Expression(("sin(x[1])", "10*x[1]*x[0]"),degree=2))
    
    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    Jhat(s)
    assert(r0>0.95)
    s.tlm_value = h
    tape.evaluate_tlm()
    r1 = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert(r1 > 1.95)
    Jhat(s)
    dJdm = Jhat.derivative().vector().inner(h.vector())
    Hm = compute_hessian(J, c, h).vector().inner(h.vector())
    r3 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert(r3 > 2.95)

