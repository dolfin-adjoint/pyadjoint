""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2

        subject to

    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega


"""
from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *

import Optizelle

parameters["adjoint"]["cache_factorizations"] = True

set_log_level(ERROR)

# Create mesh, refined in the center
n = 64
mesh = UnitSquareMesh(n, n)

cf = CellFunction("bool", mesh)
subdomain = CompiledSubDomain('std::abs(x[0]-0.5)<0.25 && std::abs(x[1]-0.5)<0.25')
subdomain.mark(cf, True)
mesh = refine(mesh, cf)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = interpolate(Expression("0.2+ 0.1*(x[0]+x[1])", degree=1), W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define functional of interest and the reduced functional
x = SpatialCoordinate(mesh)
w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3) 
d = 1/(2*pi**2)
d = Expression("d*w", d=d, w=w, degree=3) 

alpha = Constant(1e-6)
J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = Control(f)
rf = ReducedFunctional(J, control)
#rf.taylor_test(f)

# Volume constraints
class VolumeConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = volume - a*dx >= 0."""
    def __init__(self, volume, W):
        self.volume  = float(volume)

        # The derivative of the constraint g(x) is constant (it is the diagonal of the lumped mass matrix for the control function space), so let's assemble it here once.
        # This is also useful in rapidly calculating the integral each time without re-assembling.
        self.smass  = assemble(TestFunction(W) * Constant(1) * dx)
        self.tmpvec = Function(W)

    def function(self, m):
        self.tmpvec.assign(m)

        # Compute the integral of the control over the domain
        integral = self.smass.inner(self.tmpvec.vector())
        cmax = m.vector().max()
        cmin = m.vector().min()

        if MPI.rank(mpi_comm_world()) == 0:
            print("Current control integral: ", integral)
            print("Maximum of control: ", cmax)
            print("Minimum of control: ", cmin)
        return [self.volume - integral]

    def jacobian_action(self, m, dm, result):
        result[:] = self.smass.inner(-dm.vector())

    def jacobian_adjoint_action(self, m, dp, result):
        result.vector()[:] = -1.*dp[0]

    def hessian_action(self, m, dm, dp, result):
        result.vector()[:] = 0.0

    def output_workspace(self):
        return [0.0]

#problem = MinimizationProblem(rf, bounds=(0.1, 0.8), constraints=VolumeConstraint(0.3, W))
problem = MinimizationProblem(rf, bounds=(0.1, 0.8))
parameters = {
             "maximum_iterations": 100,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                 "H_type" : Optizelle.Operators.UserDefined,
                 "dir" : Optizelle.LineSearchDirection.NewtonCG,
                 #"ipm": Optizelle.InteriorPointMethod.LogBarrier,
                 "eps_grad": 1e-5,
                 "krylov_iter_max" : 40,
                 "eps_krylov" : 1e-2
                 }
             }

solver = OptizelleSolver(problem, inner_product="L2", parameters=parameters)
f_opt = solver.solve()
plot(f_opt, interactive=True)
print("Volume: ", assemble(f_opt*dx))


# Define the expressions of the analytical solution
f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w", w=w, alpha=alpha, degree=3)
u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)

# We can then compute the errors between numerical and analytical
# solutions.

f.assign(f_opt)
solve(F == 0, u, bc)
control_error = errornorm(f_analytic, f_opt)
state_error = errornorm(u_analytic, u)
print("h(min):           %e." % mesh.hmin())
print("Error in state:   %e." % state_error)
print("Error in control: %e." % control_error)
