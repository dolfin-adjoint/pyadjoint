import pytest
pytest.importorskip("firedrake")

from numpy.testing import assert_allclose
import numpy as np
from firedrake import *
from firedrake.adjoint import *
from pyadjoint import MinimizationProblem, TAOSolver
from pyadjoint.optimization.tao_solver import PETScVecInterface


def test_petsc_roundtrip_single():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)

    u_1 = Function(space_1).interpolate(X[0])
    u_2 = Function(space_2).interpolate(-1 - X[0])

    for m in [u_1, u_2]:
        vec_interface = PETScVecInterface(m)
        x = vec_interface.new_petsc()
        m_test = Function(m.function_space())
        vec_interface.to_petsc(x, m)
        vec_interface.from_petsc(x, m_test)
        assert (m.dat.data_ro == m_test.dat.data_ro).all()


def test_petsc_roundtrip_multiple():
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)

    u_1 = Function(space_1).interpolate(X[0])
    u_2 = Function(space_2).interpolate(-1 - X[0])

    vec_interface = PETScVecInterface((u_1, u_2))
    x = vec_interface.new_petsc()
    u_1_test = Function(space_1)
    u_2_test = Function(space_2)
    vec_interface.to_petsc(x, (u_1, u_2))
    vec_interface.from_petsc(x, (u_1_test, u_2_test))
    assert (u_1.dat.data_ro == u_1_test.dat.data_ro).all()
    assert (u_2.dat.data_ro == u_2_test.dat.data_ro).all()


def minimize_tao_lmvm(rf, *, convert_options=None):
    problem = MinimizationProblem(rf)
    solver = TAOSolver(problem, {"tao_type": "lmvm",
                                 "tao_gatol": 1.0e-7,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 0.0},
                       convert_options=convert_options)
    return solver.solve()


def minimize_tao_nls(rf, *, convert_options=None):
    problem = MinimizationProblem(rf)
    solver = TAOSolver(problem, {"tao_type": "nls",
                                 "tao_gatol": 1.0e-7,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 0.0},
                       convert_options=convert_options)
    return solver.solve()


@pytest.mark.parametrize("minimize", [minimize,
                                      minimize_tao_lmvm,
                                      minimize_tao_nls])
def test_optimisation_constant_control(minimize):
    """This tests a list of controls in a minimisation"""
    mesh = UnitSquareMesh(1, 1)
    R = FunctionSpace(mesh, "R", 0)

    n = 3
    x = [Function(R) for i in range(n)]
    c = [Control(xi) for xi in x]

    # Rosenbrock function https://en.wikipedia.org/wiki/Rosenbrock_function
    # with minimum at x = (1, 1, 1, ...)
    f = sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(n-1))

    J = assemble(f * dx(domain=mesh))
    rf = ReducedFunctional(J, c)
    result = minimize(rf)
    assert_allclose([float(xi) for xi in result], 1., rtol=1e-4)


def _simple_helmholz_model(V, source):
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(v), grad(u))*dx + 100.0*v*u*dx - v*source*dx
    solve(F==0, u)
    return u


def test_simple_inversion():
    """Test inversion of source term in helmholze eqn."""
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    ref = Function(V)
    source_ref = Function(V)
    x = SpatialCoordinate(mesh)
    source_ref.interpolate(cos(pi*x**2))

    # compute reference solution
    with stop_annotating():
        u_ref = _simple_helmholz_model(V, source_ref)

    # now rerun annotated model with zero source
    source = Function(V)
    c = Control(source)
    u = _simple_helmholz_model(V, source)

    J = assemble(1e6 * (u - u_ref)**2*dx)
    rf = ReducedFunctional(J, c)

    x = minimize(rf)
    assert_allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)
    rf(source)
    x = minimize(rf, derivative_options={"riesz_representation": "l2"})
    assert_allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)
    rf(source)
    x = minimize(rf, derivative_options={"riesz_representation": "H1"})
    # Assert that the optimisation does not converge for H1 representation
    assert not np.allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)


@pytest.mark.parametrize("minimize", [minimize_tao_lmvm,
                                      minimize_tao_nls])
@pytest.mark.parametrize("riesz_representation", [None, "l2", "L2", "H1"])
def test_tao_simple_inversion(minimize, riesz_representation):
    """Test inversion of source term in helmholze eqn using TAO."""
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    ref = Function(V)
    source_ref = Function(V)
    x = SpatialCoordinate(mesh)
    source_ref.interpolate(cos(pi*x**2))

    # compute reference solution
    with stop_annotating():
        u_ref = _simple_helmholz_model(V, source_ref)

    # now rerun annotated model with zero source
    source = Function(V)
    c = Control(source)
    u = _simple_helmholz_model(V, source)

    J = assemble(1e6 * (u - u_ref)**2*dx)
    rf = ReducedFunctional(J, c)

    x = minimize(rf, convert_options=(None if riesz_representation is None
                                      else {"riesz_representation": riesz_representation}))
    assert_allclose(x.dat.data, source_ref.dat.data, rtol=1e-2)


def test_tao_bounds():
    mesh = UnitIntervalMesh(11)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(space, name="u")
    u_ref = Function(space, name="u_ref").interpolate(0.5 - X[0])
    
    J = assemble((u - u_ref) ** 2 * dx)
    rf = ReducedFunctional(J, Control(u))

    lb = 0.5 - 7.0 / 11.0
    problem = MinimizationProblem(rf, bounds=(lb, None))
    solver = TAOSolver(problem, {"tao_type": "bnls",
                                 "tao_gatol": 1.0e-7,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 0.0})
    u_opt = solver.solve()

    u_ref_bound = u_ref.copy(deepcopy=True)
    u_ref_bound.dat.data[:] = np.maximum(u_ref_bound.dat.data_ro, lb)
    assert_allclose(u_opt.dat.data_ro, u_ref_bound.dat.data_ro, rtol=1.0e-2)
