from __future__ import print_function
import random

from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 3)

def main(ic, a, b, annotate=False):
    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, "-1.0", "on_boundary")

    mass = inner(u, v)*dx
    soln = Function(V)

    solve(b*mass == a*action(mass, ic), soln, bc, annotate=annotate)
    return soln

if __name__ == "__main__":

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=4), V)
    a = Constant(1.0, name="a")
    b = Constant(1.0, name="b")
    soln = main(ic, a, b, annotate=True)

    p = [Constant(2.0), Constant(3.0)]
    m = [Control(a), Control(b)]

    for (dudm, tlm_var) in compute_tlm(m, forget=False):
        # just keep iterating until we get the last dudm
        pass

    frm = inner(soln, soln)*dx
    dJdm_tlm = assemble(derivative(frm, soln)).inner(dudm.vector())

    J = Functional(frm)
    m = [ConstantControl("a"), ConstantControl("b")] # get rid of the perturbation direction \delta m

    dJdm_adm = compute_gradient(J, m, forget=False)
    dJdm_adm = float(dJdm_adm[0])*float(p[0]) + float(dJdm_adm[1])*float(p[1])

    print("dJdm_tlm: ", dJdm_tlm)
    print("dJdm_adm: ", dJdm_adm)

    assert abs(dJdm_tlm - dJdm_adm) < 1.0e-12
