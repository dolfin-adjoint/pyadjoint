""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2

    subjecct to

    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega


"""
from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *

try:
    import Optizelle
except ImportError:
    info_red("Optizelle unavailable, skipping test")
    import sys; sys.exit(0)

set_log_level(ERROR)

parameters["adjoint"]["cache_factorizations"] = True

# Create mesh
n = 8
mesh = UnitSquareMesh(n, n)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)

f = interpolate(Constant(0.11), W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define functional of interest and the reduced functional
x = SpatialCoordinate(mesh)
d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile

alpha = Constant(1e-10)
J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = Control(f)
rf = ReducedFunctional(J, control)

# Volume constraints
class VolumeConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
    def __init__(self, Vol):
        self.Vol  = float(Vol)

        # The derivative of the constraint g(x) is constant (it is the diagonal of the lumped mass matrix for the control function space), so let's assemble it here once.
        # This is also useful in rapidly calculating the integral each time without re-assembling.
        self.smass  = assemble(TestFunction(W) * Constant(1) * dx)
        self.tmpvec = Function(W, name="Control")

    def function(self, m):
        self.tmpvec.assign(m)

        # Compute the integral of the control over the domain
        integral = self.smass.inner(self.tmpvec.vector())
        vecmax   = m.vector().max()
        vecmin   = m.vector().min()
        if MPI.rank(mpi_comm_world()) == 0:
            print("Current control integral: ", integral)
            print("Maximum of control: ", vecmax)
            print("Minimum of control: ", vecmin)
        return [self.Vol - integral]

    def jacobian_action(self, m, dm, result):
        result[:] = self.smass.inner(-dm.vector())
        #print "Returning Volume Jacobian action in direction %s is %s" % (dm.vector().array(), result)

    def jacobian_adjoint_action(self, m, dp, result):
        result.vector()[:] = interpolate(Constant(-dp[0]), W).vector()

    def hessian_action(self, m, dm, dp, result):
        result.vector().zero()

    def output_workspace(self):
        return [0.0]

    def length(self):
        return 1

Vol = 0.2
Vconst = VolumeConstraint(Vol)

ub = Function(W)
ub.vector()[:] = 0.4
lb = 0.1
problem = MinimizationProblem(rf, bounds=[(lb, ub)], constraints=Vconst)

parameters = {
             "maximum_iterations": 20,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                 "H_type" : Optizelle.Operators.UserDefined,
                 "dir" : Optizelle.LineSearchDirection.NewtonCG,
                 "ipm": Optizelle.InteriorPointMethod.PrimalDualLinked,
                 "sigma": 0.001,
                 "gamma": 0.995,
                 "linesearch_iter_max" : 50,
                 "krylov_iter_max" : 100,
                 "eps_krylov" : 1e-4
                 }
             }

solver = OptizelleSolver(problem, parameters=parameters)
f_opt = solver.solve()
cmax = f_opt.vector().max()
cmin = f_opt.vector().min()

#plot(f_opt, interactive=True)

# Check that the bounds are satisfied
assert cmin >= lb
assert cmax <= ub.vector().max()
assert abs(assemble(f_opt*dx) - Vol) < 1e-3

# Check that the functional value is below the threshold
assert rf(f_opt) < 2e-4
