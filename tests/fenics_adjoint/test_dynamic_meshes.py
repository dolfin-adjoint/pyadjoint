import pytest
from dolfin import *
from dolfin_adjoint import *
import numpy as np


@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.parametrize("mesh", [UnitSquareMesh(10,10),
                                  UnitDiscMesh.create(MPI.comm_world,
                                                      10, 1, 2)])
def test_dynamic_meshes_2D(mesh, reset):
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = [Function(S), Function(S), Function(S)]
    ALE.move(mesh, s[0])

    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u0 = project(cos(pi*x[0])*sin(pi*x[1]), V)
    
    ALE.move(mesh, s[1], reset_mesh=reset)
    
    u, v = TrialFunction(V), TestFunction(V)
    f = cos(x[0]) + x[1] * sin(2 * pi * x[1])
    
    u, v = TrialFunction(V), TestFunction(V)
    dt = Constant(0.1)
    k = Constant(1/dt)
    F = k*inner(u-u0, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u1 = Function(V)
    solve(lhs(F) == rhs(F), u1)
    J = float(dt)*assemble(u1**2*dx)

    ALE.move(mesh, s[2],reset_mesh=reset)
    F = k*inner(u-u1, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u2 = Function(V)
    solve(lhs(F) == rhs(F), u2)
    J += float(dt)*assemble(u2**2*dx)
    
    ctrls = [Control(c) for c in s]
    Jhat = ReducedFunctional(J, ctrls)
    dJdm = Jhat.derivative()
        
    from pyadjoint.tape import stop_annotating
    with stop_annotating():
        A, B, C = 2, 1, 3
        taylor = [project(A*as_vector((cos(2*pi*x[1]), x[0])), S),
                  project(B*as_vector((cos(x[0]), cos(x[1]))), S),
                  project(C*as_vector((-x[0]**2, x[1])), S)]
        zero = [Function(S),Function(S), Function(S)]
        results = taylor_to_dict(Jhat, zero, taylor)
    print(results)
    assert(np.mean(results["R0"]["Rate"])>0.9)
    assert(np.mean(results["R1"]["Rate"])>1.9)
    assert(np.mean(results["R2"]["Rate"])>2.9)


@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.parametrize("mesh", [UnitCubeMesh(4,4,4),
                                  BoxMesh(Point(0,1,2),Point(1.5,2,2.5), 4,3,5)])
def test_dynamic_meshes_3D(mesh, reset):
    S = VectorFunctionSpace(mesh, "CG", 1)
    s = [Function(S), Function(S), Function(S)]
    ALE.move(mesh, s[0])

    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u0 = project(cos(pi*x[0])*sin(pi*x[1])*x[2]**2, V)
    
    ALE.move(mesh, s[1], reset_mesh=reset)
    
    u, v = TrialFunction(V), TestFunction(V)
    f = x[2]*cos(x[0]) + x[1] * sin(2 * pi * x[1])
    
    u, v = TrialFunction(V), TestFunction(V)
    dt = Constant(0.1)
    k = Constant(1/dt)
    F = k*inner(u-u0, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u1 = Function(V)
    solve(lhs(F) == rhs(F), u1)
    J = float(dt)*assemble(u1**2*dx)

    ALE.move(mesh, s[2],reset_mesh=reset)
    F = k*inner(u-u1, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u2 = Function(V)
    solve(lhs(F) == rhs(F), u2)
    J += float(dt)*assemble(u2**2*dx)
    
    ctrls = [Control(c) for c in s]
    Jhat = ReducedFunctional(J, ctrls)
    dJdm = Jhat.derivative()
        
    from pyadjoint.tape import stop_annotating
    with stop_annotating():
        taylor = [project(as_vector((sin(x[2]), cos(2*pi*x[1]),
                                     cos(x[0]*x[1]))), S),
                  project(as_vector((cos(x[0]), sin(x[2]), cos(x[1]))), S),
                  project(as_vector((cos(-x[0]**2), cos(x[2]), x[1])), S)]
        zero = [Function(S),Function(S), Function(S)]
        results = taylor_to_dict(Jhat, zero, taylor)
    print(results)
    assert(np.mean(results["R0"]["Rate"])>0.9)
    assert(np.mean(results["R1"]["Rate"])>1.9)
    assert(np.mean(results["R2"]["Rate"])>2.9)

