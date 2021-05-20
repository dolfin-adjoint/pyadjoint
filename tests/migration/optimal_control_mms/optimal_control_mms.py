""" Solves a MMS problem with smooth control """

from fenics import *
from fenics_adjoint import *

#dolfin.set_log_level(ERROR)
parameters['std_out_all_processes'] = False


def convergence_order(errors, base = 2):
    import math

    orders = [0.0] * (len(errors)-1)
    for i in range(len(errors)-1):
        try:
            orders[i] = math.log(errors[i]/errors[i+1], base)
        except ZeroDivisionError:
            orders[i] = numpy.nan

    return orders


def solve_pde(u, V, m):
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) - m*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)


def solve_optimal_control(n):
    ''' Solves the optimal control problem on a n x n fine mesh. '''

    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "DG", 0)
    m = Function(W, name='Control')

    x = SpatialCoordinate(mesh)

    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1])

    # Run the forward model once to create the annotation
    solve_pde(u, V, m)

    J = assemble((inner(u-u_d, u-u_d))*dx)

    # Run the optimisation
    control = Control(m)
    rf = ReducedFunctional(J, control)

    minimize(rf, method = "Newton-CG", tol = 1e-16, options = {'disp': True})
    solve_pde(u, V, control.tape_value())

    # Define the analytical expressions
    m_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=4)
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])", degree=4)

    # Compute the error
    control_error = errornorm(m_analytic, control.tape_value())
    state_error = errornorm(u_analytic, u)
    return control_error, state_error

try:
    control_errors = []
    state_errors = []
    element_sizes = []
    for i in range(2,6):
        n = 2**i
        control_error, state_error = solve_optimal_control(n)
        control_errors.append(control_error)
        state_errors.append(state_error)
        element_sizes.append(1./n)
        set_working_tape(Tape())

    print("Control errors: " + str(control_errors))
    print("Control convergence: " + str(convergence_order(control_errors, base = 2)))
    print("State errors: " + str(state_errors))
    print("State convergence: " + str(convergence_order(state_errors, base = 2)))

    if min(convergence_order(control_errors)) < 0.9:
        print("Convergence order below tolerance")
        sys.exit(1)
    if min(convergence_order(state_errors)) < 1.9:
        print("Convergence order below tolerance")
        sys.exit(1)
    print("Test passed")
except ImportError:
    print("No suitable scipy version found. Aborting test.")
