# .. py:currentmodule:: firedrake_adjoint
#
# Topology optimisation of fluids in Stokes flow
# ==============================================
#
# .. sectionauthor:: Patrick E. Farrell <patrick.farrell@maths.ox.ac.uk>
#
# This demo solves example 4 of :cite:`borrvall2003`.
#
# Problem definition
from firedrake import *

# ROL appears to request lots of computations whose values
# are never inspected. By default, firedrake implements
# lazy evaluation, i.e. doesn't immediately compute these
# values, but retains the computational graph that would allow
# it to do so. Unfortunately, with ROL not using its computations,
# firedrake's graph gets very large and the code spends most of
# its time updating constants. We therefore disable firedrake's
# lazy evaluation mode.
parameters["pyop2_options"]["lazy_evaluation"] = False

from firedrake.adjoint import *
from firedrake.__future__ import interpolate

try:
    import ROL
except ImportError:
    info_red("""This example depends on ROL.""")
    raise
# Starting the annotation.
continue_annotation()
mu = Constant(1.0)                   # viscosity
alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

N = 20
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
V = Constant(1.0/3) * delta  # want the fluid to occupy 1/3 of the domain

mesh = RectangleMesh(N, N, delta, 1, diagonal="right")
A = FunctionSpace(mesh, "CG", 1)        # control function space

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

(x, y) = SpatialCoordinate(mesh)
l = 1.0/6.0
gbar = 1.0
cond1 = And(gt(y, (1.0/4 - l/2)), lt(y, (1.0/4 + l/2)))
val1  = gbar*(1 - (2*(y-0.25)/l)**2)
cond2 = And(gt(y, (3.0/4 - l/2)), lt(y, (3.0/4 + l/2)))
val2  = gbar*(1 - (2*(y-0.75)/l)**2)
inflow_outflow = conditional(cond1, val1, conditional(cond2, val2, 0))


def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = Function(W)
    (u, p) = split(w)
    (v, q) = TestFunctions(W)

    F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    bcs = [DirichletBC(W.sub(0).sub(1), 0, "on_boundary"),
           DirichletBC(W.sub(0).sub(0), inflow_outflow, "on_boundary")]
    sp = {"snes_type": "ksponly",
          "snes_monitor_cancel": None,
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "mat_type": "aij"}
    solve(F == 0, w, bcs=bcs, solver_parameters=sp)

    return w

if __name__ == "__main__":
    rho = assemble(interpolate(Constant(float(V)/delta), A))
    w   = forward(rho)
    (u, p) = split(w)

    controls = File("output-rol-firedrake/control_iterations_guess.pvd")
    allctrls = File("output-rol-firedrake/allcontrols.pvd")
    rho_viz = Function(A, name="ControlVisualisation")
    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls.write(rho_viz)
        allctrls.write(rho_viz)

    J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

    # Bound constraints
    lb = 0.0
    ub = 1.0

    # We want V - \int rho dx >= 0, so write this as \int V/delta - rho dx >= 0
    volume_constraint = UFLInequalityConstraint((V/delta - rho)*dx, m)

    # Solve the optimisation problem with q = 0.01
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    params = {
        'General': {
            'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
        'Step': {
            'Type': 'Augmented Lagrangian',
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            },
            'Augmented Lagrangian': {
                'Subproblem Step Type': 'Line Search',
                'Subproblem Iteration Limit': 10
            }
        },
        'Status Test': {
            'Gradient Tolerance': 1e-7,
            'Iteration Limit': 3
        }
    }


    solver = ROLSolver(problem, params, inner_product="L2")
    rho_opt = solver.solve()
    get_working_tape().clear_tape()
    q.assign(0.1)
    rho.assign(rho_opt)
    w = forward(rho)
    (u, p) = split(w)

    # Define the reduced functionals
    controls = File("output-rol-firedrake/control_iterations_final.pvd")
    rho_viz = Function(A, name="ControlVisualisation")
    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls.write(rho_viz)
        allctrls.write(rho_viz)

    J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    params["Status Test"]["Iteration Limit"] = 15
    solver = ROLSolver(problem, params, inner_product="L2")
    rho_opt = solver.solve()
    rho_viz.assign(rho_opt)
    controls.write(rho_viz)
