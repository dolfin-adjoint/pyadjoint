from dolfin import *
from dolfin_adjoint import *
import ufl

mesh = UnitSquareMesh(3, 3)
V = FunctionSpace(mesh, "R", 0)

test = TestFunction(V)

def Fm(u, m):
    F = inner(u*u, test)*dx - inner(m, test)*dx
    return F

def main(m):
    u = interpolate(Constant(0.1), V)

    F = Fm(u, m)
    solve(F == 0, u)

    return u

def HJ(u, m, J):

    dJdu = derivative(J, u)
    F = Fm(u, m)
    dFdu = derivative(F, u)
    dFdm = ufl.algorithms.expand_derivatives(derivative(F, m))

    def HJm(mdot):
        u_tlm = Function(V)
        solve(dFdu == -action(dFdm, mdot), u_tlm)

        u_adj = Function(V)
        solve(adjoint(dFdu) == dJdu, u_adj)

        u_soa = Function(V)
        d2Jdu2 = derivative(dJdu, u, u_tlm)
        d2Fdu2 = derivative(dFdu, u, u_tlm)
        ad2Fdu2 = action(adjoint(d2Fdu2), u_adj)

        # Gah UFL this is so bloody annoying. This should Just Work
        correct_args = ufl.algorithms.extract_arguments(d2Jdu2)
        current_args = ufl.algorithms.extract_arguments(ad2Fdu2)
        ad2Fdu2 = replace(ad2Fdu2, dict(zip(current_args, correct_args)))

        # Mixed derivate term in SOA
        d2Jdudm = derivative(dJdu, m, mdot)

        solve(adjoint(dFdu) == d2Jdu2 - ad2Fdu2 + d2Jdudm, u_soa)

        # Compute the Hessian action
        dJdm = derivative(J, m)
        d2Jd2m = derivative(dJdm, m)
        # Mixed derivative term
        d2Jdmdu = derivative(dJdm, u, u_tlm)

        der = assemble(-action(adjoint(dFdm), u_soa))
        der += assemble(action(d2Jd2m, mdot))
        der += assemble(d2Jdmdu)

        return Function(V, der)

    return HJm


if __name__ == "__main__":
    m = interpolate(Constant(1), V)
    u = main(m)

    J = inner(u, u)**3*dx + inner(m, m)*dx + inner(u, m)*dx
    dJdu = derivative(J, u)

    u_adj = Function(V)
    F = Fm(u, m)
    dFdu = derivative(F, u)
    solve(adjoint(dFdu) == dJdu, u_adj)

    dFdm = ufl.algorithms.expand_derivatives(derivative(F, m))
    dJdm_vec = assemble(-action(adjoint(dFdm), u_adj)) + assemble(derivative(J, m))
    dJdm = Function(V, dJdm_vec)

    def Jhat(m):
        u = main(m)
        return assemble(inner(u, u)**3*dx + inner(m, m)*dx + inner(u, m)*dx)

    Jm = Jhat(m)

    minconv = taylor_test(Jhat, TimeConstantParameter(m), Jm, dJdm, HJm=HJ(u, m, J))
