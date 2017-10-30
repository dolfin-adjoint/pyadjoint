from __future__ import print_function
from fenics import *
from fenics_adjoint import *


mesh = UnitIntervalMesh(2)
p2 = VectorElement("CG", interval, 2)
p1 = FiniteElement("CG", interval, 1)
p2p1 = MixedElement([p2, p1])

V = FunctionSpace(mesh, p2)
P = FunctionSpace(mesh, p1)
Z = FunctionSpace(mesh, p2p1)

def main(u, p):
    assigner_u = FunctionAssigner(Z.sub(0), V)
    assigner_p = FunctionAssigner(Z.sub(1), P)

    z = Function(Z, name="Output")

    assigner_u.assign(z.sub(0), u)
    assigner_p.assign(z.sub(1), p)

    return z

if __name__ == "__main__":
    u = interpolate(Constant((1,)), V, name="Velocity")
    p = interpolate(Expression("x[0] + 1.0", degree=1), P, name="Pressure")
    z = main(u, p)

    assert adjglobals.adjointer.equation_count == 5

    success = replay_dolfin(tol=0.0, stop=True)
    assert success

    form = lambda z: inner(z, z)*dx

    J = Functional(form(z), name="a")
    m = FunctionControl("Pressure")
    Jm = assemble(form(z))
    dJdm = compute_gradient(J, m, forget=False)

    eps = 0.0001
    dJdm_fd = Function(P)
    for i in range(P.dim()):
        p_ptb = p.copy(deepcopy=True, annotate=False)
        p_vec = p_ptb.vector()
        p_vec[i] = p_vec[i][0] + eps
        z_ptb = main(u, p_ptb)
        J_ptb = assemble(form(z_ptb))
        dJdm_fd.vector()[i] = (J_ptb - Jm)/eps

    print("dJdm_fd: ", list(dJdm_fd.vector()))

    dJdm_tlm_result = Function(P)
    dJdm_tlm = compute_gradient_tlm(J, m, forget=False)
    for i in range(P.dim()):
        test_vec = Function(P)
        test_vec.vector()[i] = 1.0
        dJdm_tlm_result.vector()[i] = dJdm_tlm.inner(test_vec.vector())

    print("dJdm_tlm: ", list(dJdm_tlm_result.vector()))


    def Jhat(p):
        z = main(u, p)
        return assemble(form(z))

    minconv = taylor_test(Jhat, m, Jm, dJdm, seed=1.0e-3)
    assert minconv > 1.8

    minconv = taylor_test(Jhat, m, Jm, dJdm_tlm, seed=1.0e-3)
    assert minconv > 1.8
