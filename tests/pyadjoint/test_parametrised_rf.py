import numpy as np
import pytest
from pyadjoint.control import Control
from pyadjoint.tape import set_working_tape
from pyadjoint.reduced_functional import ParametrisedReducedFunctional, ReducedFunctional
from pyadjoint.verification import taylor_to_dict
from pyadjoint.optimization.optimization import minimize
from pyadjoint.optimization.tao_solver import MinimizationProblem, TAOSolver
from firedrake import *
from firedrake.adjoint import *
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()


# ============================================================================
#       Helper functions to build functionals with different combinations of controls and parameters
# ============================================================================
def single_control_single_param_expr(c_val, p_val):
    return c_val ** 3 * p_val

def multi_control_single_param_expr(c1_val, c2_val, p_val):
    return c1_val**3 * c2_val**4 * p_val

def single_control_multi_param_expr(c_val, p1_val, p2_val):
    return c_val**3 * p1_val * p2_val

def multi_control_multi_param_expr(c1_val, c2_val, p1_val, p2_val):
    return c1_val**3 * c2_val**4 * p1_val + c1_val**2 * c2_val**5 * p2_val

def complex_expression(c1_val, c2_val, p1_val, p2_val):
    return (c1_val + c2_val)**3 * p1_val - c1_val**2 * c2_val**2 * p2_val 

def check_taylor_test_convergence(Jhat, controls):
    """Helper function to check that the taylor test convergence rates are as expected."""
    h = [AdjFloat(1.0) for _ in controls]
    taylor_results = taylor_to_dict(Jhat, controls, h)
    assert min(taylor_results["R0"]["Rate"]) >= 0.95, f"Error in R0 rate: {taylor_results['R0']['Rate']}"
    assert min(taylor_results["R1"]["Rate"]) >= 1.95, f"Error in R1 rate: {taylor_results['R1']['Rate']}"
    assert min(taylor_results["R2"]["Rate"]) >= 2.95, f"Error in R2 rate: {taylor_results['R2']['Rate']}"

def quadratic_expression(c_val, p_val1, p_val2, p_val3):
    """A simple quadratic expression to test optimisation."""
    expression  = c_val**2 * p_val1 + c_val * p_val2 + p_val3
    optima = - p_val2 / (2 * p_val1)
    return expression, optima 

# ============================================================================
#                                                      Tests 
# ============================================================================

@pytest.mark.parametrize("c_val,p_val, mult_factor", [
    (2.0, 5.0, 1.0),
    (1.5, 3.0, 2.0),
    (4.0, 2.5, 0.5),
])
def test_parametrised_rf_basic(c_val, p_val, mult_factor):
    """Test basic evaluation of parametrised reduced functional with various values."""
    c_val = AdjFloat(c_val)
    p_val = AdjFloat(p_val)
    J = single_control_single_param_expr(c_val, p_val)
    Jhat= ParametrisedReducedFunctional(J, Control(c_val), p_val)
    
    # Test initial evaluation
    result = Jhat(c_val)
    expected = single_control_single_param_expr(c_val, p_val)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Test parameter update
    new_p = p_val * mult_factor
    Jhat.update_parameters(new_p)
    result = Jhat(c_val)
    expected = single_control_single_param_expr(c_val, new_p)
    assert result == expected
    
    # Test derivative
    check_taylor_test_convergence(Jhat, [c_val])
 


@pytest.mark.parametrize("c_val,p_val", [
    (2.0, 5.0),
    (1.0, 1.0),
    (3.5, 4.5),
])
def test_parametrised_rf_controls_property(c_val, p_val):
    """Test that controls property returns only user controls, not parameters."""
    c_val = AdjFloat(c_val)
    p_val = AdjFloat(p_val)
    J = single_control_single_param_expr(c_val, p_val)
    Jhat = ParametrisedReducedFunctional(J, Control(c_val), p_val)
    assert len(Jhat.controls) == 1
    assert Jhat.controls[0] is not None

@pytest.mark.parametrize("c_val,p1_val,p2_val,p1_new,p2_new", [
    (2.0, 3.0, 4.0, 5.0, 6.0),
    (1.5, 2.5, 3.5, 4.5, 5.5),
    (3.0, 1.0, 2.0, 3.0, 4.0),
])
def test_parametrised_rf_parameters_property(c_val, p1_val, p2_val, p1_new, p2_new):
    """Test that parameters property returns the current parameter values."""
    c_val = AdjFloat(c_val)
    p1_val = AdjFloat(p1_val)
    p2_val = AdjFloat(p2_val)
    J = single_control_multi_param_expr(c_val, p1_val, p2_val)
    Jhat = ParametrisedReducedFunctional(J, Control(c_val), parameters=[p1_val, p2_val])
    
    # Check initial parameters
    params = Jhat.parameters
    assert len(params) == 2
    assert params[0] == p1_val
    assert params[1] == p2_val
    
    # Update and check again
    Jhat.update_parameters([p1_new, p2_new])
    params = Jhat.parameters
    assert len(params) == 2
    assert params[0] == p1_new
    assert params[1] == p2_new


@pytest.mark.parametrize("c1_val,c2_val,p_val", [
    (2.0, 3.0, 5.0),
    (1.5, 2.5, 3.0),
    (4.0, 1.0, 6.0),
])
def test_parametrised_rf_call_validation(c1_val, c2_val, p_val):
    """Test that __call__ validates number of control values."""
    c1_val = AdjFloat(c1_val)
    c2_val = AdjFloat(c2_val)
    p_val = AdjFloat(p_val)

    J = multi_control_single_param_expr(c1_val, c2_val, p_val)
    Jhat = ParametrisedReducedFunctional(J, [Control(c1_val), Control(c2_val)], parameters=p_val)
    
    # Valid call
    result = Jhat([c1_val, c2_val])
    expected = multi_control_single_param_expr(c1_val, c2_val, p_val)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Invalid call - wrong number of control values
    with pytest.raises(ValueError):
        Jhat(c1_val)  # Should pass list with 2 values
    
    with pytest.raises(ValueError):
        Jhat([c1_val, c2_val, p_val])  # 3 values instead of 2

@pytest.mark.parametrize("c_val, p1_val,p2_val", [
    (2.0, 3.0, 4.0),
    (1.5, 3.5, 4.5),
    (3.0, 2.0, 3.0),
])        
def test_parametrised_rf_update_parameters_validation(c_val, p1_val, p2_val):
    """Test that update_parameters validates length of new parameters."""
    c_val = AdjFloat(c_val)
    p1_val = AdjFloat(p1_val)
    p2_val = AdjFloat(p2_val)
    J = single_control_multi_param_expr(c_val, p1_val, p2_val)
    Jhat = ParametrisedReducedFunctional(J, Control(c_val), parameters=[p1_val, p2_val])
    
    # Valid update
    Jhat.update_parameters([p1_val + 1, p2_val + 1])
    
    # Invalid update - wrong number of parameters
    with pytest.raises(ValueError):
        Jhat.update_parameters([p1_val])  # Only 1 parameter instead of 2
    
    with pytest.raises(ValueError):
        Jhat.update_parameters([p1_val, p2_val, p1_val + 1])  # 3 parameters instead of 2

def test_parametrised_rf_empty_parameter_list():
    """Test that creating a ParametrisedReducedFunctional with an empty parameter list raises an error."""
    c = AdjFloat(2.0)
    J = c * 3.0 
    with pytest.raises(ValueError):
        Jhat = ParametrisedReducedFunctional(J, Control(c), parameters=[])


@pytest.mark.parametrize("c_val,c_new,p_val,p_new", [
    (2.0, 3.0, 5.0, 7.0),
    (1.5, 2.5, 3.0, 4.5),
    (4.0, 1.0, 6.0, 2.0),
])
def test_parametrised_rf_with_single_control_single_parameter(c_val, c_new, p_val, p_new):
    """Test parametrised RF with single control and single parameter at various values."""
    c_val = AdjFloat(c_val)
    p_val = AdjFloat(p_val)
    c_new = AdjFloat(c_new)
    p_new = AdjFloat(p_new)
    J = single_control_single_param_expr(c_val, p_val)
    Jhat = ParametrisedReducedFunctional(J, Control(c_val), p_val)
    
    # Test initial evaluation
    result = Jhat(c_new)
    expected = single_control_single_param_expr(c_new, p_val)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Test derivative before update
    deriv = Jhat.derivative()
    assert len(deriv) == 1
    check_taylor_test_convergence(Jhat, [c_new])
    
    # Update parameter
    Jhat.update_parameters(p_new)
    result = Jhat(c_new)
    expected = single_control_single_param_expr(c_new, p_new)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Test derivative after update
    deriv = Jhat.derivative()
    assert len(deriv) == 1
    check_taylor_test_convergence(Jhat, [c_new])


@pytest.mark.parametrize("c1_val,c2_val, c1_new, c2_new, p_val,p_new", [
    (2.0, 3.0, 4.0, 6.0, 5.0, 7.0),
    (1.5, 2.5, 3.0, 4.5, 3.0, 4.5),
    (4.0, 1.0, 6.0, 2.0, 2.0, 3.0),
])
def test_parametrised_rf_with_multiple_controls_single_parameter(c1_val, c2_val, c1_new, c2_new, p_val, p_new):
    """Test parametrised RF with multiple controls and single parameter."""
    c1_val = AdjFloat(c1_val)
    c2_val = AdjFloat(c2_val)
    p_val = AdjFloat(p_val)
    c1_new = AdjFloat(c1_new)
    c2_new = AdjFloat(c2_new)
    p_new = AdjFloat(p_new)
    J = multi_control_single_param_expr(c1_val, c2_val, p_val)
    Jhat = ParametrisedReducedFunctional(J, [Control(c1_val), Control(c2_val)], parameters=p_val)
    
    # Test initial evaluation
    result = Jhat([c1_new, c2_new])
    expected = multi_control_single_param_expr(c1_new, c2_new, p_val)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Test initial derivatives
    derivs = Jhat.derivative()
    assert len(derivs) == 2
    check_taylor_test_convergence(Jhat, [c1_new, c2_new])


    
    # Update parameter
    Jhat.update_parameters(p_new)
    result = Jhat([c1_new, c2_new])
    expected = multi_control_single_param_expr(c1_new, c2_new, p_new)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Test derivatives after update
    derivs = Jhat.derivative()
    assert len(derivs) == 2
    check_taylor_test_convergence(Jhat, [c1_new, c2_new])


@pytest.mark.parametrize("c_val, c_new, p1_val,p2_val,p1_new,p2_new", [
    (2.0, 4.0, 3.0, 4.0, 5.0, 6.0),
    (1.5, 2.5, 3.5, 4.5, 5.5, 6.5),
    (3.0, 1.0, 2.0, 3.0, 4.0, 5.0),
])
def test_parametrised_rf_with_single_control_multiple_parameters(c_val, c_new, p1_val, p2_val, p1_new, p2_new):
    """Test parametrised RF with single control and multiple parameters."""
    c_val = AdjFloat(c_val)
    p1_val = AdjFloat(p1_val)
    p2_val = AdjFloat(p2_val)
    c_new = AdjFloat(c_new)
    p1_new = AdjFloat(p1_new)
    p2_new = AdjFloat(p2_new)
    J = single_control_multi_param_expr(c_val, p1_val, p2_val)
    Jhat = ParametrisedReducedFunctional(J, Control(c_val), parameters=[p1_val, p2_val])    
    # Test initial evaluation
    result = Jhat(c_new)
    expected = single_control_multi_param_expr(c_new, p1_val, p2_val)
    assert np.isclose(result, expected, atol=1e-8)

    # Test initial derivatives
    derivs = Jhat.derivative()
    assert len(derivs) == 1
    check_taylor_test_convergence(Jhat, [c_new])
    
    # Update parameters
    Jhat.update_parameters([p1_new, p2_new])
    result = Jhat(c_new)
    expected = single_control_multi_param_expr(c_new, p1_new, p2_new)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Test derivatives after update
    derivs = Jhat.derivative()
    assert len(derivs) == 1
    check_taylor_test_convergence(Jhat, [c_new])


@pytest.mark.parametrize("c1_val,c2_val, c1_new, c2_new, p1_val,p2_val,p1_new,p2_new", [
    (2.0, 3.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0),
    (1.5, 2.5, 3.0, 4.5, 3.0, 4.5, 4.0, 5.5),
    (3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0),
])
def test_parametrised_rf_with_multiple_controls_multiple_parameters(c1_val,c2_val,c1_new,c2_new,p1_val,p2_val,p1_new,p2_new):
    """Test parametrised RF with multiple controls and multiple parameters."""
    c1_val = AdjFloat(c1_val)
    c2_val = AdjFloat(c2_val)
    p1_val = AdjFloat(p1_val)
    p2_val = AdjFloat(p2_val)
    c1_new = AdjFloat(c1_new)
    c2_new = AdjFloat(c2_new)
    p1_new = AdjFloat(p1_new)
    p2_new = AdjFloat(p2_new)

    J = multi_control_multi_param_expr(c1_val, c2_val, p1_val, p2_val)
    Jhat = ParametrisedReducedFunctional(J, [Control(c1_val), Control(c2_val)], parameters=[p1_val, p2_val])
    
    result = Jhat([c1_new, c2_new])
    expected = multi_control_multi_param_expr(c1_new, c2_new, p1_val, p2_val)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Test newderivatives
    derivs = Jhat.derivative()
    assert len(derivs) == 2
    check_taylor_test_convergence(Jhat, [c1_new, c2_new])
    
    # Update parameters
    Jhat.update_parameters([p1_new, p2_new])
    result = Jhat([c1_new, c2_new])
    expected = multi_control_multi_param_expr(c1_new, c2_new, p1_new, p2_new)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Test derivatives after update
    derivs = Jhat.derivative()
    assert len(derivs) == 2
    check_taylor_test_convergence(Jhat, [c1_new, c2_new])


@pytest.mark.parametrize("c1_val,c2_val,c1_new, c2_new,p1_val,p2_val,p1_new,p2_new", [
    (2.0, 3.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0),
    (1.5, 2.5, 3.0, 4.5, 3.0, 4.5, 4.0, 5.5),
    (3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0),
])
def test_parametrised_rf_complex_expression(c1_val,c2_val,c1_new,c2_new,p1_val,p2_val,p1_new,p2_new):
    """Test parametrised RF with complex mathematical operations J = (c1 + c2)^2 * p1 - c1 * p2."""
    c1_val = AdjFloat(c1_val)
    c2_val = AdjFloat(c2_val)
    p1_val = AdjFloat(p1_val)
    p2_val = AdjFloat(p2_val)
    c1_new = AdjFloat(c1_new)
    c2_new = AdjFloat(c2_new)
    p1_new = AdjFloat(p1_new)
    p2_new = AdjFloat(p2_new)
    J = complex_expression(c1_val, c2_val, p1_val, p2_val)
    Jhat = ParametrisedReducedFunctional(J, [Control(c1_val), Control(c2_val)], parameters=[p1_val, p2_val])

    # Test initial evaluation
    result = Jhat([c1_new, c2_new])
    expected = complex_expression(c1_new, c2_new, p1_val, p2_val)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Update and test again
    Jhat.update_parameters([p1_new, p2_new])
    result = Jhat([c1_new, c2_new])
    expected = complex_expression(c1_new, c2_new, p1_new, p2_new)
    assert np.isclose(result, expected, atol=1e-8)

@pytest.mark.parametrize("c_val, c_new, p_val, p_new1, p_new2", [
    (2.0, 5.0, 6.0, 7.0, 8.0),
    (1.5, 3.0, 4.0, 5.0, 6.0),
    (4.0, 2.5, 3.0, 4.0, 5.0),
])
def test_parametrised_rf_multiple_update_parameters(c_val, c_new, p_val, p_new1, p_new2):
    """Test that, in case of multiple parameter updates before a call, the last update is used correctly."""
    c_val = AdjFloat(c_val)
    p_val = AdjFloat(p_val)
    c_new = AdjFloat(c_new)
    p_new1 = AdjFloat(p_new1)
    p_new2 = AdjFloat(p_new2)
    J = single_control_single_param_expr(c_val, p_val)
    Jhat = ParametrisedReducedFunctional(J, Control(c_val), parameters=p_val)
    # First update
    Jhat.update_parameters(p_new1)
    # Second update
    Jhat.update_parameters(p_new2)
    # Test evaluation uses the last updated parameter
    result = Jhat(c_new)
    expected = single_control_single_param_expr(c_new, p_new2)
    assert np.isclose(result, expected, atol=1e-8)


@pytest.mark.parametrize("c_val, c_new, p_val, p_new", [
    (2.0, 5.0, 6.0, 7.0),
    (1.5, 3.0, 4.0, 5.0),
    (4.0, 2.5, 3.0, 4.0),
])
def test_parametrised_rf_against_rf(c_val, c_new, p_val, p_new):
    """Test that the parametrised reduced functional gives the same results as a standard reduced functional with 
     derivative components."""
    # Build reduced functional with parameter as control and derivative component
    with set_working_tape() as tape_1:
        c = AdjFloat(c_val)
        p = AdjFloat(p_val)
        J = c * p
        Jhat_rf = ReducedFunctional(J, [Control(c), Control(p)], derivative_components=[0])
    
    # Build parametrised reduced functional
    with set_working_tape() as tape_2:
        J = c * p
        Jhat_param_rf = ParametrisedReducedFunctional(J, Control(c), parameters=p)
    
    # Test initial evaluation
    result_rf = Jhat_rf([c_new, p_val])
    result_param_rf = Jhat_param_rf(c_new)
    assert np.isclose(result_rf, result_param_rf, atol=1e-8)
    
    # Update parameter and test again
    result_rf_updated = Jhat_rf([c_new, p_new])
    Jhat_param_rf.update_parameters(p_new)
    result_param_rf_updated = Jhat_param_rf(c_new)
    assert np.isclose(result_rf_updated, result_param_rf_updated, atol=1e-8)

    # Test derivatives
    derivs_rf = Jhat_rf.derivative()
    derivs_param_rf = Jhat_param_rf.derivative()
    assert np.isclose(derivs_rf[0], derivs_param_rf[0], atol=1e-8)  # dJ/dc should be the same

@pytest.mark.parametrize("c_val, p_val1, p_val2, p_val3, p_val1_new, p_val2_new, p_val3_new", [
    (4.0, 3.0, 6.9, 7.4, 8.5, -3.8, 9.0),
    (5.5, 3.4, -4.0, 15.0, 9.2, 8.4, 6.7),
    (9.0, 2.5, 6.3, 1.0, 5.9, 0.0, -1.0),
])
def test_optimisation_on_quadratic_polynomial(c_val, p_val1, p_val2, p_val3, p_val1_new, p_val2_new, p_val3_new):
    """Test that we can perform an optimisation with a parametrised reduced functional on a simple quadratic polynomial."""

    c_val = AdjFloat(c_val)
    p_val1 = AdjFloat(p_val1)
    p_val2 = AdjFloat(p_val2)
    p_val3 = AdjFloat(p_val3)
    J, optima = quadratic_expression(c_val, p_val1, p_val2, p_val3)
    Jhat_prf = ParametrisedReducedFunctional(J, Control(c_val), parameters=[p_val1, p_val2, p_val3])
    
    # Perform optimisation
    result_prf = minimize(Jhat_prf)
    # Check that the optimal control value is close to the expected minimum of the quadratic
    assert np.isclose(result_prf, optima, atol=1e-6)

    # Update parameter
    p_val1_new = AdjFloat(p_val1_new)
    p_val2_new = AdjFloat(p_val2_new)
    p_val3_new = AdjFloat(p_val3_new)
    Jhat_prf.update_parameters([p_val1_new, p_val2_new, p_val3_new])
    _, new_optima = quadratic_expression(c_val, p_val1_new, p_val2_new, p_val3_new)
    # Perform optimisation
    result_prf_new = minimize(Jhat_prf)
    # Check that the optimal control value is close to the expected minimum of the quadratic
    assert np.isclose(result_prf_new, new_optima, atol=1e-6)

@pytest.mark.parametrize("c_val, p_val1, p_val2, p_val3, p_val1_new, p_val2_new, p_val3_new", [
    (4.0, 3.0, 6.9, 7.4, 8.5, -3.8, 9.0),
    (5.5, 3.4, -4.0, 15.0, 9.2, 8.4, 6.7),
    (9.0, 2.5, 6.3, 1.0, 5.9, 0.0, -1.0),
])
def test_optimisation_on_quadratic_polynomial_w_TAO(c_val, p_val1, p_val2, p_val3, p_val1_new, p_val2_new, p_val3_new):
    """Test that we can perform an optimisation with a parametrised reduced functional on a simple quadratic polynomial
        with the TAO solver."""
    n = 30
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 2)
    c_val = project(Constant(c_val), V)
    p_val1 = project(Constant(p_val1), V)
    p_val2 = project(Constant(p_val2), V)
    p_val3 = project(Constant(p_val3), V)
    p_val1_new = project(Constant(p_val1_new), V)
    p_val2_new = project(Constant(p_val2_new), V)
    p_val3_new = project(Constant(p_val3_new), V)

    J, optima = quadratic_expression(c_val, p_val1, p_val2, p_val3)
    J = assemble(J*dx)
    Jhat = ParametrisedReducedFunctional(J, Control(c_val), [p_val1, p_val2, p_val3])
    problem = MinimizationProblem(Jhat)
    parameters = { 'method': 'nls',
                   'max_it': 20,
                   'fatol' : 0.0,
                   'frtol' : 0.0,
                   'gatol' : 1e-9,
                   'grtol' : 0.0
                   }

    
    solver = TAOSolver(problem, parameters=parameters)
    m_opt = solver.solve()

    assert np.isclose(norm(m_opt-optima), 0 ,atol=1e-6)

    # Test optimisation after parameter update
    Jhat.update_parameters([p_val1_new, p_val2_new, p_val3_new])
    _, optima = quadratic_expression(c_val, p_val1_new, p_val2_new, p_val3_new)
    m_opt = solver.solve()

    assert np.isclose(norm(m_opt-optima), 0 ,atol=1e-6)