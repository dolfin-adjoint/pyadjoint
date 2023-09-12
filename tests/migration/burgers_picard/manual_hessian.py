
from dolfin import *
from dolfin_adjoint import *
import ufl_legacy as ufl
pause_annotation()

n = 2
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 1)

bcs = [DirichletBC(V, 0.0, "on_boundary")]
hbcs = [homogenize(bc) for bc in bcs]

# Work around some UFL bugs -- action(A, x) doesn't like it if A is null
ufl_action = action


def action(A, x):
    A = ufl.algorithms.expand_derivatives(A)
    if A.integrals() != ():  # form is not empty:
        return ufl_action(A, x)
    else:
        return A  # form is empty, doesn't matter anyway


def F(u, m):
    v = TestFunction(V)
    nu = Constant(0.0001)
    # Here m is the initial condition.

    F = ((u - m) * v
         + m * u.dx(0) * v
         + nu * u.dx(0) * v.dx(0)) * dx

    return F


def main(m):
    u = Function(V)
    Fm = replace(F(u, m), {u: TrialFunction(V)})
    solve(lhs(Fm) == rhs(Fm), u, bcs=bcs)
    return u


def J(u, m):
    return inner(u, u)**2 * dx


def Jhat(m):
    u = main(m)
    Jm = J(u, m)
    return assemble(Jm)


def tlm(u, m, m_dot):
    Fm = F(u, m)
    dFmdu = derivative(Fm, u)
    dFmdm = derivative(Fm, m, m_dot)
    u_tlm = Function(V)

    tlm_F = action(dFmdu, u_tlm) + dFmdm
    tlm_F = replace(tlm_F, {u_tlm: TrialFunction(V)})

    solve(lhs(tlm_F) == rhs(tlm_F), u_tlm, bcs=hbcs)
    return u_tlm


def adj(u, m):
    Fm = F(u, m)
    dFmdu = derivative(Fm, u)
    adFmdu = adjoint(dFmdu, reordered_arguments=ufl.algorithms.extract_arguments(dFmdu))

    Jm = J(u, m)
    dJdu = derivative(Jm, u, TestFunction(V))

    u_adj = Function(V)

    adj_F = action(adFmdu, u_adj) - dJdu
    adj_F = replace(adj_F, {u_adj: TrialFunction(V)})

    solve(lhs(adj_F) == rhs(adj_F), u_adj, bcs=hbcs)
    return u_adj


def dJ(u, m, u_adj):
    Fm = F(u, m)
    Jm = J(u, m)
    dFmdm = derivative(Fm, m)
    # the args argument to adjoint is the biggest time-waster ever. Everything else about the system is so beautiful :-/
    adFmdm = adjoint(dFmdm)
    current_args = ufl.algorithms.extract_arguments(adFmdm)
    correct_args = [TestFunction(V), TrialFunction(V)]
    adFmdm = replace(adFmdm, dict(list(zip(current_args, correct_args))))

    dJdm = derivative(Jm, m, TestFunction(V))

    result = assemble(-action(adFmdm, u_adj) + dJdm)
    return Function(V, result)


def soa(u, m, u_tlm, u_adj, m_dot):
    Fm = F(u, m)
    dFmdu = derivative(Fm, u)
    adFmdu = adjoint(dFmdu, reordered_arguments=ufl.algorithms.extract_arguments(dFmdu))

    dFdudu = derivative(adFmdu, u, u_tlm)
    dFdudm = derivative(adFmdu, m, m_dot)

    Jm = J(u, m)
    dJdu = derivative(Jm, u, TestFunction(V))
    dJdudu = derivative(dJdu, u, u_tlm)
    dJdudm = derivative(dJdu, m, m_dot)

    u_soa = Function(V)

    # Implement the second-order adjoint equation
    soa_F = (action(dFdudu, u_adj)
             + action(dFdudm, u_adj)
             + action(adFmdu, u_soa) +  # <-- the lhs term
             -dJdudu
             - dJdudm)
    soa_F = replace(soa_F, {u_soa: TrialFunction(V)})

    solve(lhs(soa_F) == rhs(soa_F), u_soa, bcs=hbcs)
    return u_soa


def HJ(u, m):
    def HJm(m_dot):
        u_tlm = tlm(u, m, m_dot)
        u_adj = adj(u, m)
        u_soa = soa(u, m, u_tlm, u_adj, m_dot)

        Fm = F(u, m)
        dFmdm = ufl.algorithms.expand_derivatives(derivative(Fm, m))
        adFmdm = adjoint(dFmdm)
        current_args = ufl.algorithms.extract_arguments(adFmdm)
        correct_args = [TestFunction(V), TrialFunction(V)]
        adFmdm = replace(adFmdm, dict(list(zip(current_args, correct_args))))

        Jm = J(u, m)
        dJdm = ufl.algorithms.expand_derivatives(derivative(Jm, m, TestFunction(V)))

        FH = (-action(derivative(adFmdm, u, u_tlm), u_adj) +
              -action(derivative(adFmdm, m, m_dot), u_adj) +
              -action(adFmdm, u_soa)
              + derivative(dJdm, u, u_tlm)
              + derivative(dJdm, m, m_dot))

        result = assemble(FH)
        return Function(V, result)

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
    m_dot = interpolate(Constant(1.0), V)
    seed = 0.2
    without_gradient = []
    with_gradient = []
    Jm = J_adj_m(m)
    for h in [seed * 2**-i for i in range(5)]:
        m_ptb = Function(m_dot)
        m_ptb.vector()[:] *= h
        m_tilde = Function(m)
        m_tilde.vector()[:] += m_ptb.vector()
        without_gradient.append(J_adj_m(m_tilde) - Jm)
        with_gradient.append(without_gradient[-1] - grad_J_adj_m(m, m_ptb))

    print("Taylor remainders for J(adj(m)) without gradient information: ", without_gradient)
    print("Convergence orders for above Taylor remainders: ", convergence_order(without_gradient))
    print("Taylor remainders for J(adj(m)) with gradient information: ", with_gradient)
    print("Convergence orders for above Taylor remainders: ", convergence_order(with_gradient))

    assert min(convergence_order(with_gradient)) > 1.8


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
    m_dot = interpolate(Constant(1.0), V)
    seed = 1.0
    without_gradient = []
    with_gradient = []
    Jm = Jhat(m)
    # print "m.vector(): ", m.vector().array()
    for h in [seed * 2**-i for i in range(5)]:
        m_ptb = Function(m_dot)
        m_ptb.vector()[:] *= h
        # print "m_ptb.vector(): ", m_ptb.vector().array()
        m_tilde = Function(m)
        m_tilde.vector()[:] += m_ptb.vector()
        # print "m_tilde.vector(): ", m_tilde.vector().array()
        without_gradient.append(Jhat(m_tilde) - Jm)
        correction = grad_J_u_m(m, m_ptb)
        with_gradient.append(without_gradient[-1] - correction)

    print("Taylor remainders for J(u(m)) without gradient information: ", without_gradient)
    print("Convergence orders for above Taylor remainders: ", convergence_order(without_gradient))
    print("Taylor remainders for J(u(m)) with gradient information: ", with_gradient)
    print("Convergence orders for above Taylor remainders: ", convergence_order(with_gradient))

    assert min(convergence_order(with_gradient)) > 1.8


if __name__ == "__main__":
    m = project(Constant(1.0), V)
    u = main(m)
    Jm = assemble(J(u, m))

    m_dot = interpolate(Constant(1.0), V)

    u_tlm = tlm(u, m, m_dot)
    u_adj = adj(u, m)

    dJdm = dJ(u, m, u_adj)
    # info_green("Applying Taylor test to gradient computed with adjoint ... ")
    # minconv = taylor_test(Jhat, Control(m), Jm, dJdm, value=m)
    # assert minconv > 1.8

    # info_green("Applying Taylor test to du/dm ... ")
    # little_taylor_test_dudm(m)

    # info_green("Applying Taylor test to dlambda/dm ... ")
    # little_taylor_test_dlambdadm(m)

    HJm = HJ(u, m)
    info_green("Applying Taylor test to Hessian computed with second-order adjoint ... ")
    minconv = taylor_test(Jhat, Control(m), Jm, dJdm, HJm=HJm, value=m, perturbation_direction=m_dot, seed=0.2)
    assert minconv > 2.8
