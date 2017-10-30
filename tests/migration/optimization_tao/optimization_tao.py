""" Solves a MMS problem with smooth control """
from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *
try:
    from petsc4py import PETSc
    PETSc.TAO
except Exception:
    import sys
    info_blue("PETSc bindings with TAO support unavailable, skipping test")
    sys.exit(0)


# Set options
dolfin.set_log_level(ERROR)
#dolfin.set_log_level(DBG)
parameters['std_out_all_processes'] = False
tao_args = """--petsc.tao_monitor
            --petsc.tao_view
            --petsc.tao_nls_ksp_type gltr
            --petsc.tao_nls_pc_type none
            --petsc.tao_ntr_pc_type none
           """.split()
print("Tao arguments:", tao_args)
parameters.parse(tao_args)

def solve_pde(u, V, m):
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) - m*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

if __name__ == "__main__":

    n = 100
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "DG", 0)
    m = Function(W, name='Control')

    x = SpatialCoordinate(mesh)

    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1])

    J = Functional((inner(u-u_d, u-u_d))*dx*dt[FINISH_TIME])

    # Run the forward model once to create the annotation
    solve_pde(u, V, m)

    # Run the optimisation
    rf = ReducedFunctional(J, FunctionControl(m, value=m))
    problem = MinimizationProblem(rf)
    parameters = { 'method': 'nls',
                   'max_it': 20,
                   'fatol' : 0.0,
                   'frtol' : 0.0,
                   'gatol' : 1e-9,
                   'grtol' : 0.0
                 }

    solver = TAOSolver(problem, parameters=parameters)
    m_opt = solver.solve()

    #plot(m_opt, interactive=True)

    solve_pde(u, V, m_opt)

    # Define the analytical expressions
    m_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=1)
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])", degree=1)

    # Compute the error
    control_error = errornorm(m_analytic, m_opt)
    state_error = errornorm(u_analytic, u)

    #print "Control error", control_error
    #print "State error", state_error

    # Check that values are below the threshold
    tao_p = solver.get_tao()
    assert tao_p.gnorm < 1e-9
    assert tao_p.getIterationNumber() <= 20
