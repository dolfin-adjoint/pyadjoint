# This test codes the tangent linear, first-order adjoint
# and second-order adjoints *by hand*.
# It was developed as part of the development process of the Hessian
# functionality, to build intuition.

# We're going to solve the steady Burgers' equation
# u . grad(u) - grad^2 u - f = 0
# and differentiate a functional of the solution u with respect to the
# parameter f.


from dolfin import *
from dolfin_adjoint import *
from dolfin_adjoint.utils import homogenize
import ufl.algorithms as ufl_algorithms

if dolfin.__version__ == "1.2.0":
    # work around UFL bug in dolfin 1.2.0
    expand = ufl_algorithms.expand_derivatives
else:
    expand = lambda x: x

parameters["adjoint"]["stop_annotating"] = True

mesh = UnitSquareMesh(10, 10)
Vu = VectorFunctionSpace(mesh, "CG", 2)
Vm = VectorFunctionSpace(mesh, "CG", 1)
bcs = [DirichletBC(Vu, (1.0, 1.0), "on_boundary")]
hbcs = [homogenize(bc) for bc in bcs]

# Work around some UFL bugs -- action(A, x) doesn't like it if A is null
ufl_action = action


def action(A, x):
    A = ufl_algorithms.expand_derivatives(A)
    if len(A.integrals()) != 0:  # form is not empty:
        return ufl_action(A, x)
    else:
        return A  # form is empty, doesn't matter anyway


def F(u, m):
    u_test = TestFunction(Vu)

    F = (inner(dot(grad(u), u), u_test) * dx
         + inner(grad(u), grad(u_test)) * dx +
         -inner(m, u_test) * dx)

    return F


def main(m):
    u = Function(Vu)
    Fm = F(u, m)
    solve(Fm == 0, u, J=derivative(Fm, u), bcs=bcs)
    return u


def J(u, m):
    # return inner(u, u)*dx + 0.5*inner(m, m)*dx
    return inner(u, u) * dx


def Jhat(m):
    u = main(m)
    Jm = J(u, m)
    return assemble(Jm)


def tlm(u, m, m_dot):
    Fm = F(u, m)
    dFmdu = expand(derivative(Fm, u))
    dFmdm = expand(derivative(Fm, m, m_dot))
    u_tlm = Function(Vu)

    solve(action(dFmdu, u_tlm) + dFmdm == 0, u_tlm, bcs=hbcs)
    return u_tlm


def adj(u, m):
    Fm = F(u, m)
    dFmdu = expand(derivative(Fm, u))
    adFmdu = adjoint(dFmdu, reordered_arguments=ufl_algorithms.extract_arguments(dFmdu))

    Jm = J(u, m)
    dJdu = expand(derivative(Jm, u, TestFunction(Vu)))

    u_adj = Function(Vu)

    solve(action(adFmdu, u_adj) - dJdu == 0, u_adj, bcs=hbcs)
    return u_adj


def dJ(u, m, u_adj):
    Fm = F(u, m)
    Jm = J(u, m)
    dFmdm = expand(derivative(Fm, m))
    # the args argument to adjoint is the biggest time-waster ever. Everything else about the system is so beautiful :-/
    adFmdm = adjoint(dFmdm)
    current_args = ufl_algorithms.extract_arguments(adFmdm)
    correct_args = [TestFunction(Vm), TrialFunction(Vu)]
    adFmdm = replace(adFmdm, dict(list(zip(current_args, correct_args))))

    dJdm = expand(derivative(Jm, m, TestFunction(Vm)))

    result = assemble(-action(adFmdm, u_adj) + dJdm)
    return Function(Vm, result)


def soa(u, m, u_tlm, u_adj, m_dot):
    Fm = F(u, m)
    dFmdu = expand(derivative(Fm, u))
    adFmdu = adjoint(dFmdu, reordered_arguments=ufl_algorithms.extract_arguments(dFmdu))

    dFdudu = expand(derivative(adFmdu, u, u_tlm))
    dFdudm = expand(derivative(adFmdu, m, m_dot))

    Jm = J(u, m)
    dJdu = expand(derivative(Jm, u, TestFunction(Vu)))
    dJdudu = expand(derivative(dJdu, u, u_tlm))
    dJdudm = expand(derivative(dJdu, m, m_dot))

    u_soa = Function(Vu)

    # Implement the second-order adjoint equation
    Fsoa = (action(dFdudu, u_adj)
            + action(dFdudm, u_adj)
            + action(adFmdu, u_soa)  # + # <-- the lhs term
            - dJdudu
            - dJdudm)
    solve(Fsoa == 0, u_soa, bcs=hbcs)
    return u_soa


def HJ(u, m):
    def HJm(m_dot):
        u_tlm = tlm(u, m, m_dot)
        u_adj = adj(u, m)
        u_soa = soa(u, m, u_tlm, u_adj, m_dot)

        Fm = F(u, m)
        dFmdm = ufl_algorithms.expand_derivatives(derivative(Fm, m))
        adFmdm = adjoint(dFmdm)
        current_args = ufl_algorithms.extract_arguments(adFmdm)
        correct_args = [TestFunction(Vm), TrialFunction(Vu)]
        adFmdm = replace(adFmdm, dict(list(zip(current_args, correct_args))))

        Jm = J(u, m)
        dJdm = ufl_algorithms.expand_derivatives(derivative(Jm, m, TestFunction(Vm)))

        FH = (-action(derivative(adFmdm, u, u_tlm), u_adj) +
              -action(derivative(adFmdm, m, m_dot), u_adj) +
              -action(adFmdm, u_soa)
              + derivative(dJdm, u, u_tlm)
              + derivative(dJdm, m, m_dot))

        result = assemble(FH)
        return Function(Vm, result)

    return HJm


def J_adj_m(m):
    '''J(lambda) = inner(lambda, lambda)*dx
    considered as a pure function of m
    for the purposes of Taylor verification'''
    u = main(m)
    u_adj = adj(u, m)
    return assemble(J(u_adj, m))


def grad_J_adj_m(m, m_dot):
    '''Gradient of the above function in the direction mdot.
    Correct if and only if the SOA solution is correct.'''
    u = main(m)
    u_adj = adj(u, m)
    u_tlm = tlm(u, m, m_dot)
    u_soa = soa(u, m, u_tlm, u_adj, m_dot)
    Jadj = J(u_adj, m)
    dJdadj = assemble(derivative(Jadj, u_adj))
    return dJdadj.inner(u_soa.vector())


def little_taylor_test_dlambdadm(m):
    '''Implement my own Taylor test quickly for the SOA solution.'''
    m_dot = interpolate(Constant((1.0, 1.0)), Vm)
    seed = 0.2
    without_gradient = []
    with_gradient = []
    Jm = J_adj_m(m)
    for h in [seed * 2**-i for i in range(5)]:
        m_ptb = m_dot.copy(deepcopy=True)
        m_ptb.vector()[:] *= h
        m_tilde = m.copy(deepcopy=True)
        m_tilde.vector()[:] += m_ptb.vector()
        without_gradient.append(J_adj_m(m_tilde) - Jm)
        with_gradient.append(without_gradient[-1] - grad_J_adj_m(m, m_ptb))

    print("Taylor remainders for J(adj(m)) without gradient information: ", without_gradient)
    print("Convergence orders for above Taylor remainders: ", convergence_order(without_gradient))
    print("Taylor remainders for J(adj(m)) with gradient information: ", with_gradient)
    print("Convergence orders for above Taylor remainders: ", convergence_order(with_gradient))

    assert min(convergence_order(with_gradient)) > 1.9


def grad_J_u_m(m, m_dot):
    '''Gradient of Jhat in the direction mdot, evaluated using the TLM.
    Correct if and only if the TLM solution is correct.'''
    u = main(m)
    u_tlm = tlm(u, m, m_dot)
    Jm = J(u, m)
    dJdm = assemble(derivative(Jm, u))
    dJ_tlm = dJdm.inner(u_tlm.vector())
    return dJ_tlm


def little_taylor_test_dudm(m):
    '''Implement my own Taylor test quickly for the TLM solution.'''
    m_dot = interpolate(Constant((1.0, 1.0)), Vm)
    seed = 0.2
    without_gradient = []
    with_gradient = []
    Jm = Jhat(m)
    for h in [seed * 2**-i for i in range(5)]:
        m_ptb = m_dot.copy(deepcopy=True)
        m_ptb.vector()[:] *= h
        # print "m_ptb.vector(): ", m_ptb.vector().array()
        m_tilde = m.copy(deepcopy=True)
        m_tilde.vector()[:] += m_ptb.vector()
        # print "m_tilde.vector(): ", m_tilde.vector().array()
        without_gradient.append(Jhat(m_tilde) - Jm)
        correction = grad_J_u_m(m, m_ptb)
        with_gradient.append(without_gradient[-1] - correction)

    print("Taylor remainders for J(u(m)) without gradient information: ", without_gradient)
    print("Convergence orders for above Taylor remainders: ", convergence_order(without_gradient))
    print("Taylor remainders for J(u(m)) with gradient information: ", with_gradient)
    print("Convergence orders for above Taylor remainders: ", convergence_order(with_gradient))

    assert min(convergence_order(with_gradient)) > 1.9


if __name__ == "__main__":
    # m = interpolate(Expression(("sin(x[0])", "cos(x[1])")), Vm)
    m = interpolate(Constant((2.0, 2.0)), Vm)
    u = main(m)
    Jm = assemble(J(u, m))

    m_dot = interpolate(Constant((1.0, 1.0)), Vm)

    u_tlm = tlm(u, m, m_dot)
    u_adj = adj(u, m)

    dJdm = dJ(u, m, u_adj)
    info_green("Applying Taylor test to gradient computed with adjoint ... ")
    minconv = taylor_test(Jhat, Control(m), Jm, dJdm, value=m)
    assert minconv > 1.9

    info_green("Applying Taylor test to du/dm ... ")
    little_taylor_test_dudm(m)

    info_green("Applying Taylor test to dlambda/dm ... ")
    little_taylor_test_dlambdadm(m)

    HJm = HJ(u, m)
    info_green("Applying Taylor test to Hessian computed with second-order adjoint ... ")
    minconv = taylor_test(Jhat, Control(m), Jm, dJdm, HJm=HJm, value=m, perturbation_direction=m_dot, seed=0.2)
    assert minconv > 2.9
