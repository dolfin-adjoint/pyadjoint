import random
from fenics import *
from fenics_adjoint import *
from math import sqrt

from numpy.random import rand


# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.02*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)

# Model parameters
eps    = 0.1
lmbda  = eps**2  # surface parameter
dt     = AdjFloat(5.0e-06)      # time step
theta  = 0.5          # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["std_out_all_processes"] = False;

# Create mesh and define function spaces
nodes = 32*32
mesh = UnitSquareMesh(int(sqrt(nodes)), int(sqrt(nodes)))
cg1 = FiniteElement("Lagrange", triangle, 1)
ME = FunctionSpace(mesh, MixedElement([cg1, cg1]))

steps = 5

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
    u.assign(ic, annotate=annotate)
    u0.assign(ic, annotate=annotate)

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

    file = File("output.pvd", "compressed")

    # Step in time
    t = 0.0
    T = steps*dt
    j = 0.5 * dt * assemble((1.0/(4*eps)) * (pow( (-1.0/eps) * u0[1], 2))*dx)

    while (t < T):
        t += dt
        solve(L == 0, u, J=a, solver_parameters=parameters, annotate=annotate)

        u0.assign(u, annotate=annotate)

        file << (u.split()[0], t)
        if t >= T:
            quad_weight = 0.5
        else:
            quad_weight = 1.0
        j += quad_weight * dt * assemble((1.0/(4*eps)) * (pow( (-1.0/eps) * u0[1], 2))*dx)

    return u0, j

if __name__ == "__main__":
    ic = interpolate(InitialConditions(degree=1), ME)

    forward, j = main(ic, annotate=True)

    J = j
    c = Control(ic)
    dJdic = compute_gradient(J, c)

    h = Function(ME)
    h.vector()[:] = 0.1*rand(ME.dim())
    dJdic = h._ad_dot(dJdic)

    Hic = compute_hessian(J, c, h)
    Hic = h._ad_dot(Hic)

    def J(ic):
        u, j = main(ic, annotate=False)
        return j

    minconv = taylor_test(J, ic, h, dJdic, Hm=Hic)
    assert minconv > 2.8
