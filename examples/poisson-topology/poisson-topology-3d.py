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
# This demo solves a 3D generalisation of example 1 of :cite:`gersborg2006`.
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
# where :math:`\Omega` is the unit cube, :math:`T` is the temperature,
# :math:`a` is the control (:math:`a(x) = 1` means material, :math:`a(x)
# = 0` means no material), :math:`f` is a prescribed source term (here
# the constant :math:`10^{-2}`), :math:`k(a)` is the Solid Isotropic
# Material with Penalisation parameterisation :cite:`bendsoe2003`
# :math:`\epsilon + (1 - \epsilon) a^p` with :math:`\epsilon` and
# :math:`p` prescribed constants, :math:`\alpha` is a regularisation
# term, and :math:`V` is the volume bound on the control.
#
# Physically, this corresponds to finding the material distribution
# :math:`a(x)` that produces the least heat when the amount of high
# conduction material is limited.
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

# Next, we set some PETSc options to govern how the linear systems
# are to be solved: this sets the number of smoother interations
# in the smoothed aggregation multigrid.

PETScOptions.set("pc_gamg_agg_nsmooths", 1)

# Next we import the Python interface to IPOPT. If IPOPT is
# unavailable on your system, we strongly :doc:`suggest you install it
# <../../download/index>`; IPOPT is a well-established open-source
# optimisation algorithm.

try:
    from pyadjoint import ipopt
except ImportError:
    from ufl_legacy.log import info_red

    info_red("""This example depends on IPOPT and Python ipopt bindings. \
  When compiling IPOPT, make sure to link against HSL, as it \
  is a necessity for practical problems.""")
    raise

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

# Next we define some constants, and the Solid Isotropic Material with
# Penalisation (SIMP) rule.

V = Constant(0.4, name="Control")  # volume bound on the control
p = Constant(5)  # power used in the solid isotropic material
# with penalisation (SIMP) rule, to encourage the control solution to attain either 0 or 1
eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material
# with penalisation (SIMP) rule, used to encourage the control solution to attain either 0 or 1
alpha = Constant(1.0e-8)  # regularisation coefficient in functional


def k(a):
    """Solid isotropic material with penalisation (SIMP) conductivity
  rule, equation (11)."""
    return eps + (1 - eps) * a ** p


# Next we define the mesh (a unit square) and the function spaces to be
# used for the control :math:`a` and forward solution :math:`T`.

n = 10
mesh = UnitCubeMesh(n, n, n)
A = FunctionSpace(mesh, "CG", 1)  # function space for control
P = FunctionSpace(mesh, "CG", 1)  # function space for solution


# Next we define the forward boundary condition and source term.

class DirichletBoundary(SubDomain):
    """
    The left, top and far boundaries of the unit cube, used to enforce the
    Dirichlet boundary condition.
    """

    def inside(self, x, on_boundary):
        return (x[0] == 0.0 or x[1] == 1.0 or x[2] == 1.0) and on_boundary


# Define the Dirichlet BC; the Neumann BC will be implemented implicitly by
# dropping the surface integral after integration by parts

bc = [DirichletBC(P, 0.0, DirichletBoundary())]
f = interpolate(Constant(1.0e-2), P)  # the volume source term for the PDE


# Next we define a function that given a control :math:`a` solves the
# forward PDE for the temperature :math:`T`. (The advantage of
# formulating it in this manner is that it makes it easy to conduct
# :doc:`Taylor remainder convergence tests
# <../../documentation/verification>`.)

def forward(a):
    """Solve the forward problem for a given material distribution a(x)."""
    T = Function(P, name="Temperature")
    v = TestFunction(P)

    F = inner(grad(v), k(a) * grad(T)) * dx - f * v * dx
    solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7,
                                                              "maximum_iterations": 20,
                                                              "linear_solver": "cg",
                                                              "preconditioner": "petsc_amg"}})

    return T


# Now we define the ``__main__`` section. We define the initial guess
# for the control and use it to solve the forward PDE. In order to
# ensure feasibility of the initial control guess, we interpolate the
# volume bound; this ensures that the integral constraint and the
# bound constraint are satisfied.

if __name__ == "__main__":
    a = interpolate(V, A)  # initial guess.
    T = forward(a)  # solve the forward problem once.

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

    controls = File("output-3d/control_iterations.pvd")
    a_viz = Function(A, name="ControlVisualisation")

    def eval_cb(j, a):
        a_viz.assign(a)
        controls << a_viz

    # Now we define the functional, compliance with a weak regularisation
    # term on the gradient of the material

    J = assemble(f * T * dx + alpha * inner(grad(a), grad(a)) * dx)
    m = Control(a)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

    # This :py:class:`ReducedFunctional` object solves the forward PDE using
    # dolfin-adjoint's tape each time the functional is to be evaluated, and
    # derives and solves the adjoint equation each time the functional
    # gradient is to be evaluated. The :py:class:`ReducedFunctional` object
    # takes in high-level Dolfin objects (i.e. the input to the evaluation
    # ``Jhat(a)`` would be a :py:class:`dolfin.Function`).

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

    class VolumeConstraint(InequalityConstraint):
        """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""

        def __init__(self, V):
            self.V = float(V)

            # The derivative of the constraint g(x) is constant (it is the
            # diagonal of the lumped mass matrix for the control function space),
            # so let's assemble it here once.  This is also useful in rapidly
            # calculating the integral each time without re-assembling.

            self.smass = assemble(TestFunction(A) * Constant(1) * dx)
            self.tmpvec = Function(A)

        def function(self, m):
            from pyadjoint.reduced_functional_numpy import set_local
            set_local(self.tmpvec, m)

            # Compute the integral of the control over the domain

            integral = self.smass.inner(self.tmpvec.vector())
            if MPI.rank(MPI.comm_world) == 0:
                print("Current control integral: ", integral)
            return [self.V - integral]

        def jacobian(self, m):
            return [-self.smass]

        def output_workspace(self):
            return [0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 1

    # Now that all the ingredients are in place, we can perform the
    # optimisation.  The :py:class:`MinimizationProblem` class
    # represents the optimisation problem to be solved. We instantiate
    # this and pass it to :py:mod:`ipopt` to solve:

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=VolumeConstraint(V))
    parameters = None

    solver = IPOPTSolver(problem, parameters=parameters)
    a_opt = solver.solve()

    infile = XDMFFile(MPI.comm_world, 'output-3d/control_solution.xdmf')
    infile.write(a_opt)

# The example code can be found in ``examples/poisson-topology/`` in the
# ``dolfin-adjoint`` source tree. Running it takes approximately 5 minutes
# on my laptop. The code is executed as follows:

# .. code-block:: bash

#   $ mpiexec -n 4 python poisson-topology.py
#   ...

#   Number of Iterations....: 17
#
#   (scaled)                 (unscaled)
#   Objective...............:   1.0060497104608291e-04    1.0060497104608291e-04
#   Dual infeasibility......:   1.3469311941544299e-08    1.3469311941544299e-08
#   Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
#   Complementarity.........:   1.3512182623547745e-09    1.3512182623547745e-09
#   Overall NLP error.......:   1.3469311941544299e-08    1.3469311941544299e-08
#
#
#   Number of objective function evaluations             = 18
#   Number of objective gradient evaluations             = 18
#   Number of equality constraint evaluations            = 0
#   Number of inequality constraint evaluations          = 18
#   Number of equality constraint Jacobian evaluations   = 0
#   Number of inequality constraint Jacobian evaluations = 18
#   Number of Lagrangian Hessian evaluations             = 0
#   Total CPU secs in IPOPT (w/o function evaluations)   =     30.014
#   Total CPU secs in NLP function evaluations           =    253.252
#
#   EXIT: Solved To Acceptable Level.

# The optimisation iterations can be visualised by opening
# ``output/control_iterations.pvd`` in paraview. The resulting solution
# exhibits fascinating structures.

# .. image:: poisson-topology-3d.png
#     :scale: 90
#     :align: center

# .. rubric:: References

# .. bibliography:: /documentation/poisson-topology/poisson-topology.bib
#    :cited:
#    :labelprefix: 3E-
