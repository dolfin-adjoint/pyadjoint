from dolfin import *
from dolfin_adjoint import *
import sys

parameters["adjoint"]["fussy_replay"] = True

adj_checkpointing(strategy='multistage', steps=4,
                  snaps_on_disk=5, snaps_in_ram=10, verbose=True)

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)

def main(nu):
    u = ic.copy(deepcopy=True)
    u_next = Function(V)
    v = TestFunction(V)

    timestep = Constant(1.0/n)

    F = ((u_next - u)/timestep*v
        + u_next*u_next.dx(0)*v
        + nu*u_next.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.1
    while (t <= end):
        solve(F == 0, u_next, bc)
        u.assign(u_next)
        t += float(timestep)
        adj_inc_timestep()

    return u

if __name__ == "__main__":
    nu = Constant(0.0001)
    u = main(nu)

    adj_html("forward.html", "forward")
    adj_html("adjoint.html", "adjoint")

    adj_check_checkpoints()

    J = Functional(inner(u,u)*dx*dt[FINISH_TIME])
    dJdnu = compute_gradient(J, Control(nu))

    parameters["adjoint"]["stop_annotating"] = True

    Jnu = assemble(inner(u, u)*dx) # current value

    def Jhat(nu): # the functional as a pure function of nu
        u = main(nu)
        return assemble(inner(u, u)*dx)

    conv_rate = taylor_test(Jhat, Control(nu), Jnu, dJdnu)

    if conv_rate < 1.9:
        sys.exit(1)
