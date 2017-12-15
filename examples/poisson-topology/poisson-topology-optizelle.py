#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _poisson-topology-example:
#
# .. py:currentmodule:: dolfin_adjoint
#
# Topology optimisation of heat conduction problems governed by the Poisson equation
# ==================================================================================
#
# .. sectionauthor:: Patrick E. Farrell <patrick.farrell@maths.ox.ac.uk>
#
# This demo solves example 1 of :cite:`gersborg2006`.
#
# Problem definition
# ******************
#
# This problem is to minimise the compliance
#
# .. math::
#       \int_{\Omega} fT + \alpha \int_{\Omega} \nabla a \cdot \nabla a
#
# subject to the Poisson equation with mixed Dirichlet--Neumann
# conditions
#
# .. math::
#       -\mathrm{div}(k(a) \nabla T) &= f  \qquad \mathrm{in} \ \Omega           \\
#                         T &= 0  \qquad \mathrm{on} \ \delta \Omega_D  \\
#           (k(a) \nabla T) &= 0  \qquad \mathrm{on} \ \delta \Omega_N  \\
#
# and to the control constraints
#
# .. math::
#          0 \le a(x) &\le 1  \qquad \forall x \in \Omega \\
#          \int_{\Omega} a &\le V
#
# where :math:`\Omega` is the unit square, :math:`T` is the temperature,
# :math:`a` is the control (:math:`a(x) = 1` means material, :math:`a(x)
# = 0` means no material), :math:`f` is a prescribed source term (here
# the constant :math:`10^{-2}`), :math:`k(a)` is the Solid Isotropic
# Material with Penalisation parameterisation :cite:`bendsoe2003`
# :math:`\epsilon + (1 - \epsilon) a^p` with :math:`\epsilon` and
# :math:`p` prescribed constants, :math:`\alpha` is a regularisation
# term, and :math:`V` is the volume bound on the control.
#
# Physically, the problem is to finding the material distribution
# :math:`a(x)` that minimises the integral of the temperature when the amount of highly
# conducting material is limited. This code makes several approximations to
# this physical problem. Instead of solving an integer optimisation problem (at each
# location, we either have conducting material or we do not), a continuous relaxation
# is performed; this is standard in topology optimisation :cite:`bendsoe2003`. Furthermore,
# the discrete solution varies as the mesh is refined: the continuous solution exhibits
# features at all scales, and these must be carefully handled in a discretisation
# of the problem. In this example we merely add a fixed :math:`H^1` regularisation
# term; a better approach is to add a mesh-dependent Helmholtz filter (see for example
# :cite:`lazarov2011`).
#
# This example demonstrates how to implement general control
# constraints, and how to use IPOPT :cite:`wachter2006` to solve the
# optimisation problem.
#
# Implementation
# **************
#
# First, the :py:mod:`dolfin` and :py:mod:`dolfin_adjoint` modules are
# imported:

from dolfin import *
from dolfin_adjoint import *
set_log_level(WARNING)

# Next we import the Python interface to IPOPT. If IPOPT is
# unavailable on your system, we strongly :doc:`suggest you install it
# <../../download/index>`; IPOPT is a well-established open-source
# optimisation algorithm.

import Optizelle

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

# Next we define some constants, and the Solid Isotropic Material with
# Penalisation (SIMP) rule.

V = Constant(0.4)      # volume bound on the control
p = Constant(5)        # power used in the solid isotropic material
                       # with penalisation (SIMP) rule, to encourage the control
                       # solution to attain either 0 or 1
eps = Constant(1.0e-3) # epsilon used in the solid isotropic material
alpha = Constant(1.0e-8) # regularisation coefficient in functional


def k(a):
    """Solid isotropic material with penalisation (SIMP) conductivity
  rule, equation (11)."""
    return eps + (1 - eps) * a**p

# Next we define the mesh (a unit square) and the function spaces to be
# used for the control :math:`a` and forward solution :math:`T`.

n = 100
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "CG", 1)  # function space for control
P = FunctionSpace(mesh, "CG", 1)  # function space for solution

# Next we define the forward boundary condition and source term.

class WestNorth(SubDomain):
    """The top and left boundary of the unitsquare, used to enforce the Dirichlet boundary condition."""
    def inside(self, x, on_boundary):
        return (x[0] == 0.0 or x[1] == 1.0) and on_boundary

# the Dirichlet BC; the Neumann BC will be implemented implicitly by
# dropping the surface integral after integration by parts
bc = [DirichletBC(P, 0.0, WestNorth())]
f = interpolate(Constant(1.0e-2), P, name="SourceTerm") # the volume source term for the PDE

# Next we define a function that given a control :math:`a` solves the
# forward PDE for the temperature :math:`T`. (The advantage of
# formulating it in this manner is that it makes it easy to conduct
# :doc:`Taylor remainder convergence tests
# <../../documentation/verification>`.)


def forward(a):
    """Solve the forward problem for a given material distribution a(x)."""
    T = Function(P, name="Temperature")
    v = TestFunction(P)

    F = inner(grad(v), k(a)*grad(T))*dx - f*v*dx
    solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7,
                                                              "maximum_iterations": 20}})

    return T

# Now we define the ``__main__`` section. We define the initial guess
# for the control and use it to solve the forward PDE. In order to
# ensure feasibility of the initial control guess, we interpolate the
# volume bound; this ensures that the integral constraint and the
# bound constraint are satisfied.

if __name__ == "__main__":
    a = interpolate(Constant(0.3), A, name="Control") # initial guess.
    T = forward(a)                        # solve the forward problem once.

# With the forward problem solved once, :py:mod:`dolfin_adjoint` has
# built a *tape* of the forward model; it will use this tape to drive
# the optimisation, by repeatedly solving the forward model and the
# adjoint model for varying control inputs.

# A common task when solving optimisation problems is to implement a
# callback that gets executed at every functional evaluation. (For
# example, this might be to record the value of the functional so that
# it can be plotted as a function of iteration, or to record statistics
# about the controls suggested by the optimisation algorithm.) The
# following callback outputs each evaluation to VTK format, for
# visualisation in paraview. Note that the callback will output each
# *evaluation*; this means that it will be called more often than the
# number of iterations the optimisation algorithm reports, due to line
# searches. It is also possible to implement :doc:`callbacks that are
# executed on every functional derivative calculation
# <../../documentation/optimisation>`.

    controls = File("output/control_iterations.pvd")
    a_viz = Function(A, name="ControlVisualisation")
    def eval_cb(j, a):
        a_viz.assign(a)
        controls << a_viz

# Now we define the functional, compliance with a weak regularisation
# term on the gradient of the material

    J = Functional(f*T*dx + alpha * inner(grad(a), grad(a))*dx)
    m = Control(a)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

# This :py:class:`ReducedFunctional` object solves the forward PDE using
# dolfin-adjoint's tape each time the functional is to be evaluated, and
# derives and solves the adjoint equation each time the functional
# gradient is to be evaluated. The :py:class:`ReducedFunctional` object
# takes in high-level Dolfin objects (i.e. the input to the evaluation
# ``Jhat(a)`` would be a :py:class:`dolfin.Function`). But optimisation
# algorithms expect to pass :py:mod:`numpy` arrays in and
# out. Therefore, we introduce a :py:class:`ReducedFunctionalNumPy`
# class that wraps the :py:class:`ReducedFunctional` to handle array
# input and output.

    rfn  = ReducedFunctionalNumPy(Jhat)

# Now let us configure the control constraints. The bound constraints
# are easy:

    lb = 0.0
    ub = 1.0

# The volume constraint involves a little bit more work. Following
# :cite:`nocedal2006`, inequality constraints are represented as
# (possibly vector) functions :math:`g` defined such that :math:`g(a)
# \ge 0`. The constraint is implemented by subclassing the
# :py:class:`InequalityConstraint` class. (To implement equality
# constraints, see the documentation for
# :py:class:`EqualityConstraint`.)  In this case, our :math:`g(a) = V -
# \int_{\Omega} a`. In order to implement the constraint, we have to
# implement three methods: one to compute the constraint value, one to
# compute its Jacobian, and one to return the number of components in
# the constraint.

    out = File("output/iterations.pvd")
    # Volume constraints
    class VolumeConstraint(InequalityConstraint):
        """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
        def __init__(self, V):
            self.V  = float(V)
            self.scale = 1.

# The derivative of the constraint g(x) is constant (it is the
# diagonal of the lumped mass matrix for the control function space),
# so let's assemble it here once.  This is also useful in rapidly
# calculating the integral each time without re-assembling.

            self.smass  = assemble(TestFunction(A) * Constant(1) * dx)
            self.tmpvec = Function(A)

        def function(self, m):
            self.tmpvec.assign(m)
            out << self.tmpvec

            # Compute the integral of the control over the domain
            integral = self.smass.inner(self.tmpvec.vector())
            return [self.scale * (self.V - integral)]

        def jacobian_action(self, m, dm, result):
            result[:] = - self.scale * self.smass.inner(dm.vector())

        def jacobian_adjoint_action(self, m, dp, result):
            result.vector()[:] = interpolate(Constant(-self.scale * dp[0]), A).vector()

        def hessian_action(self, m, dm, dp, result):
            result.vector().zero()

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1

# Now that all the ingredients are in place, we can perform the
# optimisation.  The :py:class:`ReducedFunctionalNumPy` class has a
# method :py:meth:`ReducedFunctionalNumPy.pyipopt_problem`, which
# creates a :py:class:`pyipopt.Problem` class that represents the
# optimisation problem to be solved. We call this and pass it to
# :py:mod:`pyipopt` to solve:

    parameters["adjoint"]["stop_annotating"] = True
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(V))

    parameters = {
                 "maximum_iterations": 200,
                 "optizelle_parameters":
                     {
                     "msg_level" : 10,
                     "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                     "H_type" : Optizelle.Operators.ScaledIdentity,
                     "dir" : Optizelle.LineSearchDirection.BFGS,
                     "eps_dx": 1.0e-32,
                     "linesearch_iter_max" : 5,
                     "ipm": Optizelle.InteriorPointMethod.PrimalDual,
                     "mu": 1e-5,
                     "eps_mu": 1e-3,
                     "sigma" : 0.5,
                     "delta" : 100.,
                     #"xi_qn" : 1e-12,
                     #"xi_pg" : 1e-12,
                     #"xi_proj" : 1e-12,
                     #"xi_tang" : 1e-12,
                     #"xi_lmh" : 1e-12,
                     "rho" : 100.,
                     #"augsys_iter_max" : 1000,
                     "krylov_iter_max" : 40,
                     "eps_krylov" : 1e-6,
                     #"dscheme": Optizelle.DiagnosticScheme.DiagnosticsOnly,
                     "h_diag" : Optizelle.FunctionDiagnostics.SecondOrder,
                     "x_diag" : Optizelle.VectorSpaceDiagnostics.Basic,
                     "y_diag" : Optizelle.VectorSpaceDiagnostics.Basic,
                     "z_diag" : Optizelle.VectorSpaceDiagnostics.EuclideanJordan,
                     #"f_diag" : Optizelle.FunctionDiagnostics.FirstOrder,
                     "g_diag" : Optizelle.FunctionDiagnostics.SecondOrder,
                     "L_diag" : Optizelle.FunctionDiagnostics.SecondOrder,
                     "stored_history": 25,
                     }
                 }

    solver  = OptizelleSolver(problem, parameters=parameters)
    a_opt   = solver.solve()
    xdmf_filename = XDMFFile(mpi_comm_world(), "output/control_solution.xdmf")
    xdmf_filename.write(a_opt)
