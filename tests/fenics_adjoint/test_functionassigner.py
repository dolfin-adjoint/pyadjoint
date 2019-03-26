from dolfin import *
from dolfin_adjoint import *
import numpy as np

def test_function_assigner_poisson():
    mesh = UnitSquareMesh(15,15)
    CG1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    R = FiniteElement("R", mesh.ufl_cell(), 0)
    VR = FunctionSpace(mesh, MixedElement([CG1,R]))
    V = FunctionSpace(mesh, CG1)

    S = VectorFunctionSpace(mesh, "CG", 1)
    s_ = Function(S)
    ALE.move(mesh, s_)

    u, r = TrialFunctions(VR)
    v, s = TestFunctions(VR)

    ur = Function(VR, name="(u,r)")

    x = SpatialCoordinate(mesh)
    f = cos(2*pi*x[0])*cos(2*pi*x[1])
    a = inner(grad(u), grad(v))*dx
    a += inner(r, v)*dx + inner(u, s)*dx
    A = assemble(a)
    l = inner(f, v)*dx
    L = assemble(l)

    solve(A, ur.vector(), L)
    uh = Function(V, name="uh")
    R_space = FunctionSpace(mesh, R)
    rh = Function(R_space)
    fa = FunctionAssigner([V, R_space] , VR)
    fa.assign([uh, rh],ur)
    J = assemble(uh*ds) + assemble(uh*uh**2*dx)# + assemble(inner(f,f)*dx)

    Jhat = ReducedFunctional(J, Control(s_))
    dJ1 = Jhat.derivative()

    from pyadjoint.tape import stop_annotating
    with stop_annotating():
        A = 1
        pert = project(A*Expression(("x[0]","cos(pi*x[1])"),degree=3), S)
        results = taylor_to_dict(Jhat, s_, pert)
        assert(min(results["FD"]["Rate"]) > 0.95)
        assert(min(results["dJdm"]["Rate"]) > 1.95)
        assert(min(results["Hm"]["Rate"]) > 2.95)

    tape = get_working_tape()
    tape.reset_tlm_values()
    s_.tlm_value = pert
    tape.evaluate_tlm()
    r1_tlm = taylor_test(Jhat, s_, pert, dJdm=J.block_variable.tlm_value)
    assert(r1_tlm > 1.95)
    Jhat(s_)
    # Solve same problem with split
    uh, rh = ur.split()
    J = assemble(uh*ds) + assemble(uh*uh**2*dx)

    Jhat = ReducedFunctional(J, Control(s_))
    dJ2 = Jhat.derivative()
    assert(np.allclose(dJ1.vector().get_local(), dJ2.vector().get_local()))
if __name__ == "__main__":
     test_function_assigner_poisson()
