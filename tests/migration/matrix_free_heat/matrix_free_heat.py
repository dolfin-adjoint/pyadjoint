import sys

from dolfin import *
from dolfin_adjoint import *

dolfin.parameters["adjoint"]["record_all"] = True

f = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)", degree=4)
mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)

def run_forward(initial_condition=None, annotate=True, dump=True):
    u = TrialFunction(V)
    v = TestFunction(V)

    u_0 = Function(V)
    if initial_condition is not None:
        u_0.assign(initial_condition)

    u_1 = Function(V)

    dt = Constant(0.1)

    F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx

    bc = DirichletBC(V, 1.0, "on_boundary")

    a, L = lhs(F), rhs(F)

    solver = AdjointPETScKrylovSolver("default", "none")
    matfree = AdjointKrylovMatrix(a, bcs=bc)

    t = float(dt)
    T = 1.0
    n = 1

    if dump:
        u_out = File("u.pvd", "compressed")
        u_out << u_0

    while t <= T:
        b_rhs = assemble(L)
        bc.apply(b_rhs)
        solver.solve(matfree, down_cast(u_0.vector()), down_cast(b_rhs), annotate=annotate)

        t += float(dt)
        if dump:
            u_out << u_0

    return u_0

if __name__ == "__main__":

    final_forward = run_forward()

    adj_html("heat_forward.html", "forward")
    adj_html("heat_adjoint.html", "adjoint")

    # The functional is only a function of final state.
    functional=Functional(final_forward*final_forward*dx*dt[FINISH_TIME])
    dJdic = compute_gradient(functional, InitialConditionParameter(final_forward), forget=False)

    def J(ic):
        perturbed_u0 = run_forward(initial_condition=ic, annotate=False, dump=False)
        return assemble(perturbed_u0*perturbed_u0*dx)

    minconv = utils.test_initial_condition_adjoint(J, Function(V), dJdic, seed=10.0)

    if minconv < 1.9:
        sys.exit(1)

    dJ = assemble(derivative(final_forward*final_forward*dx, final_forward))

    ic = final_forward
    ic.vector()[:] = 0

    minconv = utils.test_initial_condition_tlm(J, dJ, ic, seed=10.0)

    if minconv < 1.9:
        sys.exit(1)
