import sys
from numpy.random import random
from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 3)
dolfin.parameters["adjoint"]["record_all"] = True

def main(ic, annotate=False):
    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, "-1.0", "on_boundary")

    mass = inner(u, v)*dx
    soln = Function(V)

    solve(mass == action(mass, ic), soln, bc, annotate=annotate)
    return soln

if __name__ == "__main__":

    ic = project(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=4), V, name="InitialCondition")
    soln = main(ic, annotate=True)
    parameters["adjoint"]["stop_annotating"] = True

    perturbation_direction = Function(V)
    vec = perturbation_direction.vector()
    vec_size = vec.local_size()
    vec.set_local(random(vec_size))
    vec.apply("")

    m = Control(ic, perturbation=perturbation_direction, value=ic)
    Jm = assemble(soln*soln*dx)
    J = Functional(soln*soln*dx)
    dJdm = compute_gradient_tlm(J, m, forget=False)

    def Jhat(ic):
        soln = main(ic, annotate=False)
        return assemble(soln*soln*dx)

    minconv = taylor_test(Jhat, m, Jm, dJdm)
    assert minconv > 1.9

    adj_reset()
