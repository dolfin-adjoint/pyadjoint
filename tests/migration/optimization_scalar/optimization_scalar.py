from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *
import sys

dolfin.set_log_level(ERROR)

n = 10
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
u = ic.copy(deepcopy=True)

def main(nu):
    u_next = Function(V)
    v = TestFunction(V)

    timestep = Constant(1.0/n, name="Timestep")

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

def eval_cb(j, m):
    print("j = %f, m = %f." % (j, float(m)))

def derivative_cb(j, dj, m):
    print("j = %f, dj = %f, m = %f." % (j, dj, float(m)))

def replay_cb(var, data, m):
    #print "Got data for variable %s at m = %f." % (var, float(m))
    pass

if __name__ == "__main__":
    nu = Constant(0.0001, name="Nu")
    # Run the forward model once to have the annotation
    main(nu)

    J = Functional(inner(u, u)*dx*dt[FINISH_TIME])

    # Run the optimisation
    reduced_functional = ReducedFunctional(J, ConstantControl("Nu"),
                                           eval_cb_post= eval_cb,
                                           derivative_cb_post=derivative_cb,
                                           replay_cb=replay_cb,
                                           scale=2.0)
    try:
        nu_opt = minimize(reduced_functional, 'SLSQP')

        tol = 1e-4
        if reduced_functional(nu_opt) > tol:
            print('Test failed: Optimised functional value exceeds tolerance: ', reduced_functional(nu_opt), ' > ', tol, '.')
            sys.exit(1)
    except ImportError:
        info_red("No suitable scipy version found. Aborting test.")
