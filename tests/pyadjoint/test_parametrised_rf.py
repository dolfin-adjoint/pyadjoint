import pytest
from pyadjoint import *
from pyadjoint.control import Control
from pyadjoint.tape import set_working_tape
from pyadjoint.reduced_functional import ParametrisedReducedFunctional, ReducedFunctional
from pyadjoint.verification import taylor_to_dict
import numpy as np


# ============================================================================
# Helper functions to build functionals with different combinations of controls and parameters
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

# ============================================================================
# Tests 
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