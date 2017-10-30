""" Solves a MMS problem with smooth control """
from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *
try:
    from pyOpt.pySNOPT import SNOPT
except ImportError:
    import sys
    info_blue("pyopt bindings unavailable, skipping test")
    sys.exit(0)


dolfin.set_log_level(INFO)
parameters['std_out_all_processes'] = False

def solve_pde(u, V, m):
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) - m*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

def solve_optimal_control(n):

    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "DG", 0)
    m = Function(W, name='Control')
    x = SpatialCoordinate(mesh)

    x = SpatialCoordinate(mesh)

    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1])

    J = Functional((inner(u-u_d, u-u_d))*dx*dt[FINISH_TIME])

    # Run the forward model once to create the annotation
    solve_pde(u, V, m)

    # Run the optimisation
    rf = ReducedFunctional(J, Control(m, value=m))

    rfn = ReducedFunctionalNumPy(rf)
    nlp, grad = rfn.pyopt_problem()
    snopt = SNOPT(options={"Major feasibility tolerance": 1e-10,
                           "Major optimality tolerance": 1e-10,
                           "Minor feasibility tolerance": 1e-10,
                           "Major print level": 1,
                           "Minor print level": 1})
    res = snopt(nlp, sens_type=grad)
    print(snopt.getInform(res[-1]["value"]))
    m.vector()[:] = res[1]

    #plot(m, interactive=True)

    solve_pde(u, V, m)


    # Define the analytical expressions
    m_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=4)
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])", degree=4)

    # Compute the error
    control_error = errornorm(m_analytic, m)
    state_error = errornorm(u_analytic, u)
    return control_error, state_error

control_error, state_error = solve_optimal_control(n=50)
print("Error in control: ", control_error)
print("Error in state: ", state_error)

assert state_error < 0.0002
