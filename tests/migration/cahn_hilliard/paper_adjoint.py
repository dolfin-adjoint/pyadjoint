from __future__ import print_function
import random
import sys
from datetime import datetime

from dolfin import *
from dolfin_adjoint import *
from math import sqrt

dolfin.parameters["adjoint"]["fussy_replay"] = False

# Class representing the intial conditions
class InitialConditions(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)

# Model parameters
eps    = 0.1
lmbda  = eps**2  # surface parameter
dt     = 5.0e-06      # time step
theta  = 0.5          # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
parameters["std_out_all_processes"] = False;

# Create mesh and define function spaces
nodes = 500000
#nodes = 1000
mesh = UnitSquareMesh(int(sqrt(nodes)), int(sqrt(nodes)))
V = FunctionSpace(mesh, "Lagrange", 1)
ME = V*V

steps = 50
adj_checkpointing('multistage', steps=steps+1, snaps_on_disk=10, snaps_in_ram=5, verbose=True)

def main(ic, annotate=False):

    # Define trial and test functions
    du    = TrialFunction(ME)
    q, v  = TestFunctions(ME)

    # Define functions
    u   = Function(ME, name="NextSolution")  # current solution
    u0  = Function(ME, name="Solution")  # solution from previous converged step

    # Split mixed functions
    dc, dmu = split(du)
    c,  mu  = split(u)
    c0, mu0 = split(u0)

    # Create intial conditions and interpolate
    u.assign(ic, annotate=False)
    u0.assign(ic, annotate=False)

    # Compute the chemical potential df/dc
    c = variable(c)
    f    = 100*c**2*(1-c)**2
    dfdc = diff(f, c)

    # mu_(n+theta)
    mu_mid = (1.0-theta)*mu0 + theta*mu

    # Weak statement of the equations
    L0 = c*q*dx - c0*q*dx + dt*dot(grad(mu_mid), grad(q))*dx
    L1 = mu*v*dx - dfdc*v*dx - lmbda*dot(grad(c), grad(v))*dx
    L = L0 + L1

    # Compute directional derivative about u in the direction of du (Jacobian)
    a = derivative(L, u, du)

    # Create nonlinear problem and Newton solver
    parameters = {}
    parameters["newton_solver"] = {}
    parameters["newton_solver"]["convergence_criterion"] = "incremental"
    parameters["newton_solver"]["relative_tolerance"] = 1e-6

    if annotate:
        file = File("output.pvd", "compressed")

    # Step in time
    t = 0.0
    T = steps*dt
    import os
    j = 0.5 * dt * assemble((1.0/(4*eps)) * (pow( (-1.0/eps) * u0[1], 2))*dx)

    if annotate:
        adjointer.time.start(t)

    while (t < T):
        t += dt
        print("Starting solve at t=%s: " % t, datetime.now().isoformat())
        solve(L == 0, u, J=a, solver_parameters=parameters, annotate=annotate)
        print("Finished solve at t=%s: " % t, datetime.now().isoformat())

        if annotate:
            file << (u.split()[0], t)

        u0.assign(u, annotate=annotate)

        if t >= T:
            quad_weight = 0.5
        else:
            quad_weight = 1.0
        j += quad_weight * dt * assemble((1.0/(4*eps)) * (pow( (-1.0/eps) * u0[1], 2))*dx)

        if annotate:
            adj_inc_timestep(time=t, finished=t>=T)

    return u0, j

if __name__ == "__main__":
    ic = Function(ME)
    init = InitialConditions()
    ic.interpolate(init)
    ic_copy = Function(ic)
    tlm_copy = Function(ic)

    timer = Timer("Original forward run")
    forward, j = main(ic, annotate=True)
    ic = forward
    ic.vector()[:] = ic_copy.vector()
    timer.stop()
    forward_copy = Function(forward)

    adj_html("forward.html", "forward")

    dtm = TimeMeasure()
    J = Functional((1.0/(4*eps)) * (pow( (-1.0/eps) * forward[1], 2))*dx*dtm)
    dJdic = compute_gradient(J, InitialConditionParameter(ic))

    def J(ic):
        u, j = main(ic, annotate=False)
        return j

    minconv = utils.test_initial_condition_adjoint(J, ic_copy, dJdic, seed=1.0e-7)
    if minconv < 1.9:
        sys.exit(1)
