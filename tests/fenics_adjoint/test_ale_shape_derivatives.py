import numpy as np
from dolfin import *
from dolfin_adjoint import *



def test_sin_weak():
    mesh = UnitSquareMesh(6,6)
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

def test_sin_weak_spatial():
    mesh = UnitDiscMesh.create(MPI.comm_world, 10, 1, 2)
    X = SpatialCoordinate(mesh)

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



def test_tlm_assemble():
    mesh = UnitCubeMesh(4,4,4)
    X = SpatialCoordinate(mesh)

    tape = get_working_tape()
    tape.clear_tape()
    set_working_tape(tape)
    S =  VectorFunctionSpace(mesh, "CG", 1)
    h = Function(S)
    h.interpolate(Expression(("A*cos(x[1])","A*x[2]", "A*x[1]"),degree=2,A=2.5))

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
    s.block_variable.tlm_value = h
    tape = get_working_tape()
    tape.evaluate_tlm()
    r1_tlm = taylor_test(Jhat, s, h, dJdm=J.block_variable.tlm_value)
    assert(r1_tlm > 1.9)
    Jhat(s)
    r1 = taylor_test(Jhat, s, h)
    assert(np.isclose(r1,r1_tlm, rtol=1e-14))


def test_shape_hessian():
    mesh = SphericalShellMesh.create(MPI.comm_world, 1)
    X = SpatialCoordinate(mesh)
    tape = get_working_tape()
    tape.clear_tape()
    X = SpatialCoordinate(mesh)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")

    ALE.move(mesh, s)
    integrand = X[2]**2*cos(X[1])**2
    J = assemble(integrand* dx(domain=mesh))
    c = Control(s)
    Jhat = ReducedFunctional(J, c)

    h = Function(S,name="V")
    h.interpolate(Expression(("A*cos(x[1])","B*x[2]", "C*x[1]"),degree=2,A=10, B=5, C=1.2))


    # Second order taylor
    dJdm = Jhat.derivative().vector().inner(h.vector())
    Hm = compute_hessian(J, c, h).vector().inner(h.vector())
    r2 = taylor_test(Jhat, s, h, dJdm=dJdm, Hm=Hm)
    assert(r2 > 2.9)
    Jhat(s)
    dJdmm_exact = derivative(derivative(integrand* dx(domain=mesh),X,h), X, h)
    assert(np.isclose(assemble(dJdmm_exact), Hm))


def test_PDE_hessian():
    tape = get_working_tape()
    tape.clear_tape()
    mesh = UnitSquareMesh(6,6)

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
    A = 5
    h = project(as_vector((X[0], cos(pi/3*X[1],))), S)

    # Finite difference
    r0 = taylor_test(Jhat, s, h, dJdm=0)
    Jhat(s)
    assert(r0>0.95)
    # First order taylor
    s.block_variable.tlm_value = h
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


def test_repeated_movement():
    mesh = UnitIntervalMesh(10)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s0 = Function(S)
    ALE.move(mesh, s0)
    ALE.move(mesh, s0)

    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u0 = project(cos(pi*x[0]), V)

    u, v = TrialFunction(V), TestFunction(V)
    f = cos(x[0]) + x[0] * sin(2 * pi * x[0])

    u, v = TrialFunction(V), TestFunction(V)
    dt = Constant(0.1)
    k = Constant(1/dt)
    F = k*inner(u-u0, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u1 = Function(V)
    solve(lhs(F) == rhs(F), u1)
    J = assemble(u1**2*dx)

    c = Control(s0)
    Jhat = ReducedFunctional(J, c)
    dJdm = Jhat.derivative()


    taylor = project(as_vector((cos(x[0]),)), S)
    zero = Function(S)
    results = taylor_to_dict(Jhat, zero, taylor)
    assert(np.mean(results["R0"]["Rate"])>0.9)
    assert(np.mean(results["R1"]["Rate"])>1.9)
    assert(np.mean(results["R2"]["Rate"])>2.9)

    tape = get_working_tape()
    tape.clear_tape()

    mesh = UnitIntervalMesh(10)
    S = VectorFunctionSpace(mesh, "CG", 1)
    s0 = Function(S)
    c = Control(s0)
    s0.assign(Constant(2)*s0)
    ALE.move(mesh, s0)

    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u0 = project(cos(pi*x[0]), V)

    u, v = TrialFunction(V), TestFunction(V)
    f = cos(x[0]) + x[0] * sin(2 * pi * x[0])

    u, v = TrialFunction(V), TestFunction(V)
    dt = Constant(0.1)
    k = Constant(1/dt)
    F = k*inner(u-u0, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u1 = Function(V)
    solve(lhs(F) == rhs(F), u1)
    J = assemble(u1**2*dx)

    Jhat = ReducedFunctional(J, c)
    dJdm_2 = Jhat.derivative()
    assert(np.allclose(dJdm.vector().get_local(),
                       dJdm_2.vector().get_local()))

    taylor = project(as_vector((x[0]-3,)), S)
    zero = Function(S)
    results = taylor_to_dict(Jhat, zero, taylor)
    assert(np.mean(results["R0"]["Rate"])>0.9)
    assert(np.mean(results["R1"]["Rate"])>1.9)
    assert(np.mean(results["R2"]["Rate"])>2.9)
