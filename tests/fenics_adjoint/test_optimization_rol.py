from __future__ import print_function
import pytest
import numpy

pytest.importorskip("fenics")
pytest.importorskip("ROL")

from fenics import *
from fenics_adjoint import *


@pytest.fixture
def setup():
    set_log_level(ERROR)

    n = 20
    mesh = UnitSquareMesh(n, n)

    cf = CellFunction("bool", mesh)
    subdomain = CompiledSubDomain(
        'std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
    subdomain.mark(cf, True)
    mesh = refine(mesh, cf)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 0)

    f = interpolate(Expression("x[0]+x[1]", degree=1), W)
    u = Function(V, name='State')
    v = TestFunction(V)

    F = (inner(grad(u), grad(v)) - f*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

    w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    d = 1/(2*pi**2)
    d = Expression("d*w", d=d, w=w, degree=3)

    alpha = Constant(1e-3)
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
    control = Control(f)

    rf = ReducedFunctional(J, control)
    params = {
        'Step': {
            'Type': 'Line Search',
        },
        'Status Test': {
            'Gradient Tolerance': 1e-11,
            'Iteration Limit': 20
        }
    }
    return rf, params, w, alpha


def test_finds_analytical_solution(setup):
    rf, params, w, alpha = setup
    problem = MinimizationProblem(rf)
    solver = ROLSolver(problem, params, inner_product="L2")
    sol = solver.solve()
    f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w",
                            w=w, alpha=alpha, degree=3)

    assert errornorm(f_analytic, sol) < 0.02


def test_bounds_work_sensibly(setup):
    rf, params, w, alpha = setup
    lower = 0
    upper = 0.5

    problem = MinimizationProblem(rf, bounds=(lower, upper))
    solver = ROLSolver(problem, params, inner_product="L2")
    sol1 = solver.solve().copy(deepcopy=True)
    f = rf.controls[0]
    V = f.function_space()

    lower = interpolate(Constant(lower), V)
    upper = interpolate(Constant(upper), V)
    problem = MinimizationProblem(rf, bounds=(lower, upper))
    solver = ROLSolver(problem, params, inner_product="L2")
    solver.rolvector.scale(0.0)
    sol2 = solver.solve()

    assert errornorm(sol1, sol2) < 1e-8
