from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *
import numpy as np
import os, sys


# Set log level
set_log_level(LogLevel.WARNING)

# Prepare a mesh
mesh = UnitIntervalMesh(50)

# Choose a time step size
k = Constant(1e-3)

# Compile sub domains for boundaries
left  = CompiledSubDomain("near(x[0], 0.)")
right = CompiledSubDomain("near(x[0], 1.)")

# Label boundaries, required for the objective
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
left.mark(boundary_parts, 0)
right.mark(boundary_parts, 1)
ds = Measure("ds", subdomain_data=boundary_parts)


class Source(Expression):
    def __init__(self, omega=Constant(2e2), **kwargs):
        """ Construct the source function """
        self.t = 0.0
        self.omega = omega

    def eval(self, value, x):
        """ Evaluate the source function and it's derivative"""
        if x[0] < 1e-15:
            value[0] = np.sin(float(self.omega) * self.t)
        else:
            value[0] = 0.


class SourceDerivative(Expression):
    def __init__(self, omega=Constant(2e2), Source=None, **kwargs):
        """ Construct the source function """
        super().__init__(**kwargs)
        self.t = 0.0
        self.omega = omega
        self.source = Source  # needed to get the matching time instant

    def eval(self, value, x):
        """ Evaluate the source function and it's derivative"""
        if x[0] < 1e-15:
            value[0] = self.source.t * np.cos(float(self.omega) * self.source.t)
        else:
            value[0] = 0.

def forward(excitation, c=Constant(1.), record=False, annotate=False, objective=None):
    """ The forward problem """
    
    # Define function space
    U = FunctionSpace(mesh, "Lagrange", 1)

    # Set up initial values
    u0 = Function(U, name = "u0", annotate = annotate)
    u1 = Function(U, name = "u1", annotate = annotate)

    # Define test and trial functions
    v = TestFunction(U)
    u = TrialFunction(U)

    # Define variational formulation
    udot = (u - 2.*u1 + u0)
    uold = (0.25*u + 0.5*u1 +0.25*u0)
    F = (udot*v+k*k*c*c*uold.dx(0)*v.dx(0))*dx - u*v*ds(0) + excitation*v*ds(0)
    a = lhs(F)
    L = rhs(F)

    # Prepare solution
    u = Function(U, name = "u", annotate = annotate)

    # The actual timestepping
    if record: rec = [u1(1.),]
    i = 1
    t = 0.0        # Initial time
    T = 3.e-1      # Final time
    times = [t,]
    objective(u1, times[-1])
    while t < T - .5*float(k):
        excitation.t = t + float(k)
        solve(a == L, u, annotate = annotate)
        u0.assign(u1, annotate = annotate)
        u1.assign(u, annotate = annotate)

        t = i*float(k)
        times.append(t)
        if record:
            rec.append(u1(1.0))
        if annotate: adj_inc_timestep(t, t > T - .5*float(k))
        i += 1
        if objective is not None:
            objective(u1, times[-1])

    if record:
        np.savetxt("recorded.txt", rec)

    return u1, times

# Callback function for the optimizer
# Writes intermediate results to a logfile
def eval_cb(j, m):
    """ The callback function keeping a log """

    print("omega = %15.10e " % float(m))
    print("objective = %15.10e " % j)

def optimize(dbg=False):
    """ The optimization routine """
    
    # Define the control
    Omega = Constant(190)
    source = Source(Omega, degree=3, name="source")
    
    # provide the coefficient on which this expression depends and its derivative
    source.dependencies = [Omega]
    source.user_defined_derivatives = {Omega: SourceDerivative(Omega, Source = source, degree=3)}

    if dbg:
        # Check the recorded tape
        success = replay_dolfin(tol = 0.0, stop = True)
        print("replay: ", success)

        # for the equations recorded on the forward run
        adj_html("forward.html", "forward")
        # for the equations to be assembled on the adjoint run
        adj_html("adjoint.html", "adjoint")

    # Load references
    refs = np.loadtxt("recorded.txt")

    # create noise to references
    gamma = 1.e-5
    if gamma > 0:
        noise = np.random.normal(0, gamma, refs.shape[0])

        # add noise to the refs
        refs += noise

    # map refs to be constant
    refs = list(map(Constant, refs))

    J = {"functional": 0, "refs": refs, "refs_idx": 0, "first_time": None, "last_time": None}
    def objective(u, t, retrieve_value=False):
        if retrieve_value:
            area = J["last_time"] - J["first_time"]
            M = J["refs_idx"]
            return area/M*J["functional"]
        obs = J["refs"][J["refs_idx"]]
        J["functional"] += assemble(inner(u - obs, u - obs)*ds(1))
        J["refs_idx"] += 1
        J["last_time"] = t
        if J["first_time"] is None:
            J["first_time"] = t

    # Execute first time to annotate and record the tape
    u, times = forward(source, 2*DOLFIN_PI, False, True, objective=objective)

    # Define the control
    control = Control(Omega)

    J = objective(None, None, True)
    
    # compute the gradient
    dJd0 = compute_gradient(J, control)
    print("gradient = ", float(dJd0))
    
    # Prepare the reduced functional
    reduced_functional = ReducedFunctional(J, control, eval_cb_post = eval_cb)

    # Run the optimisation
    omega_opt = minimize(reduced_functional, method = "L-BFGS-B",
                         tol=1.0e-12, options = {"disp": True,"gtol":1.0e-12})

    # Print the obtained optimal value for the controls
    print("omega = %f" %float(omega_opt))

if __name__ == "__main__":
    if '-r' in sys.argv:
        os.popen('rm -rf recorded.txt')
        source = Source(Constant(2e2), degree=3)
        forward(source, 2*DOLFIN_PI, True)

    if '-dbg' in sys.argv:
        optimize(True)
    else:
        optimize()
