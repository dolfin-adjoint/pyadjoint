""" Solves an optimisation problem with the Burgers equation as constraint """

from __future__ import print_function
import sys

from dolfin import *
from dolfin_adjoint import *
import scipy
import libadjoint

dolfin.set_log_level(ERROR)

n = 20
end = 0.2
timestep = Constant(1.0/n)
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

adj_checkpointing(strategy='multistage',
                  steps=int(end*n)+1,
                  snaps_on_disk=2,
                  snaps_in_ram=2,
                  verbose=True)

def Dt(u_next, u, timestep):
    return (u_next - u)/timestep

def main(u, annotate=False):

    u_next = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    F = (Dt(u_next, u, timestep)*v
         + u_next*u_next.dx(0)*v + nu*u_next.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    adjointer.time.start(t)
    while (t <= end):
        solve(F == 0, u_next, bc, annotate=annotate)
        u.assign(u_next, annotate=annotate)

        t += float(timestep)
        adj_inc_timestep(time=t, finished = t>end)

def derivative_cb(j, dj, m):
    print("j = %f, max(dj) = %f, max(m) = %f." % (j, dj.vector().max(), m.vector().max()))

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V, annotate=False)
    u = ic.copy(deepcopy=True, name='Velocity')

    J = Functional(u*u*dx*dt[FINISH_TIME])

    # Run the model once to create the annotation
    u.assign(ic, annotate=False)
    main(u, annotate=True)

    # Run the optimisation
    lb = interpolate(Constant(-1),  V)

    # Define the reduced funtional
    reduced_functional = ReducedFunctional(J, Control(u, value=ic),
            derivative_cb_post=derivative_cb)

    print("\n === Solving problem with L-BFGS-B. === \n")
    try:
        u_opt = minimize(reduced_functional, method = 'L-BFGS-B', bounds = (lb, 1), tol = 1e-10, options = {'disp': True})
    except ImportError:
        info_red("No suitable scipy version found. Aborting test.")
        from sys import exit; exit(0)

    tol = 1e-9
    final_functional = reduced_functional(u_opt)
    print("Final functional value: ", final_functional)
    if final_functional > tol:
        print('Test failed: Optimised functional value exceeds tolerance: ' , final_functional, ' > ', tol, '.')
        sys.exit(1)

    # Run the problem again with SQP

    # Method specific arguments:
    options = {"SLSQP": {"bounds": (lb, 1)},
               "BFGS": {"bounds": None},
               "COBYLA": {"bounds": None, "rhobeg": 0.1},
               "TNC": {"bounds": None},
               "L-BFGS-B": {"bounds": (lb, 1)},
               "Newton-CG": {"bounds": None, "maxiter": 1},
               "Nelder-Mead": {"bounds": None },
               "Anneal": {"bounds": None, "lower": -0.1, "upper": 0.1},
               "CG": {"bounds": None},
               "Powell": {"bounds": None}
              }

    for method in ["SLSQP", "BFGS", "COBYLA", "TNC", "L-BFGS-B", "Nelder-Mead", "CG"]: #, "Powell"]:
        print("\n === Solving problem with %s. ===\n" % method)
        u_opt.assign(ic, annotate = False)
        reduced_functional(u_opt)
        u_opt = minimize(reduced_functional,
                         bounds = options[method].pop("bounds"),
                         method = method, tol = 1e-10,
                         options = dict({'disp': True, "maxiter": 2}, **options[method]))
    info_green("Test passed")
