from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *

if not hasattr(dolfin, "FunctionAssigner"):
    info_red("Need dolfin.FunctionAssigner for this test.")
    import sys
    sys.exit(0)

mesh = UnitIntervalMesh(2)
cg2 = VectorElement("CG", interval, 2)
cg1 = FiniteElement("CG", interval, 1)
ele = MixedElement([cg2, cg1])
V = FunctionSpace(mesh, cg2)
Z = FunctionSpace(mesh, ele)

def main(z0):
    """ Maps
       [ v ]
       [ p ] ->  [v]
    """
    assigner_u = FunctionAssigner(V, Z.sub(0))

    v = Function(V, name="Output")

    assigner_u.assign(v, z0.sub(0))

    return v

if __name__ == "__main__":
    z0 = interpolate(Constant((1, 2)), Z, name="State")
    v = main(z0)

    parameters["adjoint"]["stop_annotating"] = True

    # Check that the function assignment worked
    assert tuple(v.vector()) == (1, 1, 1, 1, 1)

    # Check taht the annotation worked
    assert adjglobals.adjointer.equation_count == 3

    success = replay_dolfin(tol=0.0, stop=True)
    assert success

    form = lambda v: inner(v, v)*dx

    J = Functional(form(v), name="a")
    m = FunctionControl("State")
    Jm = assemble(form(v))
    dJdm = compute_gradient(J, m, forget=False)

    eps = 0.0001
    dJdm_fd = Function(Z)
    for i in range(Z.dim()):
        z_ptb = z0.copy(deepcopy=True)
        vec = z_ptb.vector()
        vec[i] = vec[i][0] + eps
        v_ptb = main(z_ptb)
        J_ptb = assemble(form(v_ptb))
        dJdm_fd.vector()[i] = (J_ptb - Jm)/eps

    print("dJdm_fd: ", list(dJdm_fd.vector()))

    dJdm_tlm_result = Function(Z)
    dJdm_tlm = compute_gradient_tlm(J, m, forget=False)
    for i in range(Z.dim()):
        test_vec = Function(Z)
        test_vec.vector()[i] = 1.0
        dJdm_tlm_result.vector()[i] = dJdm_tlm.inner(test_vec.vector())

    print("dJdm_tlm: ", list(dJdm_tlm_result.vector()))



    def Jhat(z):
        v = main(z)
        return assemble(form(v))

    minconv = taylor_test(Jhat, m, Jm, dJdm, seed=1.0e-3)
    assert minconv > 1.8

    minconv = taylor_test(Jhat, m, Jm, dJdm_tlm, seed=1.0e-3)
    assert minconv > 1.8
