""" Runs various diagnostic tests provided by Optizelle. """

from dolfin import *
from dolfin_adjoint import *
try:
    import Optizelle
except ImportError:
    info_blue("Optizelle bindings unavailable, skipping test")
    sys.exit(0)

set_log_level(WARNING)

V = Constant(0.4)      # volume bound on the control
p = Constant(5)        # power used in the solid isotropic material
                       # with penalisation (SIMP) rule, to encourage the control
                       # solution to attain either 0 or 1
eps = Constant(1.0e-3) # epsilon used in the solid isotropic material
alpha = Constant(1.0e-8) # regularisation coefficient in functional

def k(a):
    return eps + (1 - eps) * a**p

n = 10
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "CG", 1)  # function space for control
P = FunctionSpace(mesh, "CG", 1)  # function space for solution

class WestNorth(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] == 0.0 or x[1] == 1.0) and on_boundary
bc = [DirichletBC(P, 0.0, WestNorth())]
f = interpolate(Constant(1.0e-2), P, name="SourceTerm") # the volume source term for the PDE


def forward(a):
    T = Function(P)
    v = TestFunction(P)

    F = inner(grad(v), k(a)*grad(T))*dx - f*v*dx
    solve(F == 0, T, bc)
    return T

a = interpolate(Constant(0.3), A, name="Control")
T = forward(a)

J = Functional(f*T*dx + alpha * inner(grad(a), grad(a))*dx)
m = Control(a)
Jhat = ReducedFunctional(J, m)

lb = 0.0
ub = 1.0

class VolumeConstraint(EqualityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
    def __init__(self, V):
        self.V  = float(V)

        self.smass  = assemble(TestFunction(A) * Constant(1) * dx)
        self.tmpvec = Function(A)

    def function(self, m):
        self.tmpvec.assign(m)

        # Compute the integral of the control over the domain
        integral = self.smass.inner(self.tmpvec.vector())
        return [self.V - integral]

    def jacobian_action(self, m, dm, result):
        result[:] = - self.smass.inner(dm.vector())

    def jacobian_adjoint_action(self, m, dp, result):
        result.vector()[:] = interpolate(Constant(-dp[0]), A).vector()

    def hessian_action(self, m, dm, dp, result):
        result.vector().zero()

    def output_workspace(self):
        return [0.0]

    def length(self):
        return 1

# Instantiate and rescale the bound constraints manually,
# so that the finite difference test reaches the region of convergence
lower_bound = OptizelleBoundConstraint(m.data(), 0., 'lower')
lower_bound.scale *= 1000.
upper_bound = OptizelleBoundConstraint(m.data(), 1., 'upper')
upper_bound.scale *= 1000.

problem = MinimizationProblem(Jhat,
        constraints=[lower_bound, upper_bound, VolumeConstraint(V)])
parameters = {
             "maximum_iterations": 200,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                 "H_type" : Optizelle.Operators.SR1,
                 "dir" : Optizelle.LineSearchDirection.BFGS,
                 "eps_dx": 1.0e-32,
                 "linesearch_iter_max" : 5,
                 "mu": 1e+0,
                 "eps_mu": 1e-6,
                 "krylov_iter_max" : 30,
                 "sigma" : 0.5,
                 "dscheme": Optizelle.DiagnosticScheme.DiagnosticsOnly,
                 "h_diag" : Optizelle.FunctionDiagnostics.SecondOrder,
                 "x_diag" : Optizelle.VectorSpaceDiagnostics.Basic,
                 "y_diag" : Optizelle.VectorSpaceDiagnostics.Basic,
                 "z_diag" : Optizelle.VectorSpaceDiagnostics.EuclideanJordan,
                 "f_diag" : Optizelle.FunctionDiagnostics.SecondOrder,
                 "g_diag" : Optizelle.FunctionDiagnostics.SecondOrder,
                 "L_diag" : Optizelle.FunctionDiagnostics.SecondOrder,
                 "stored_history": 25,
                 }
             }

solver  = OptizelleSolver(problem, parameters=parameters)
a_opt   = solver.solve()
