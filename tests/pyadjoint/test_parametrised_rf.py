import pytest
from pyadjoint import *
from pyadjoint.control import Control
from pyadjoint.tape import set_working_tape
from pyadjoint.reduced_functional import ParametrisedReducedFunctional, ReducedFunctional


# ============================================================================
# Helper functions to build functionals with different combinations of controls and parameters
# ============================================================================

def build_single_control_single_param(c_val, p_val):
    """Build J(c, p) = c * p with single control and parameter.
    
    Returns:
        tuple: (Jhat, c_val, p_val)
    """
    c = AdjFloat(c_val)
    p = AdjFloat(p_val)
    J = c * p
    Jhat = ParametrisedReducedFunctional(J, Control(c), p)
    return Jhat


def build_multi_control_single_param(c1_val, c2_val, p_val):
    """Build J(c1, c2, p) = c1 * c2 * p with multiple controls and single parameter.
    
    Returns:
        tuple: (Jhat, c1_val, c2_val, p_val)
    """
    c1 = AdjFloat(c1_val)
    c2 = AdjFloat(c2_val)
    p = AdjFloat(p_val)
    J = c1 * c2 * p
    Jhat = ParametrisedReducedFunctional(J, [Control(c1), Control(c2)], p)
    return Jhat

def build_single_control_multi_param(c_val, p1_val, p2_val):
    """Build J(c, p1, p2) = c * p1 * p2 with single control and multiple parameters.
    
    Returns:
        tuple: (Jhat, c_val, p1_val, p2_val)
    """
    c = AdjFloat(c_val)
    p1 = AdjFloat(p1_val)
    p2 = AdjFloat(p2_val)
    J = c * p1 * p2
    Jhat = ParametrisedReducedFunctional(J, Control(c), [p1, p2])
    return Jhat


def build_multi_control_multi_param(c1_val, c2_val, p1_val, p2_val):
    """Build J(c1, c2, p1, p2) = c1^2 * c2 * p1 + c2^2 * p2.
    
    Returns:
        tuple: (Jhat, c1_val, c2_val, p1_val, p2_val)
    """
    c1 = AdjFloat(c1_val)
    c2 = AdjFloat(c2_val)
    p1 = AdjFloat(p1_val)
    p2 = AdjFloat(p2_val)
    J = c1 * c1 * c2 * p1 + c2 * c2 * p2
    Jhat = ParametrisedReducedFunctional(J, [Control(c1), Control(c2)], [p1, p2])
    return Jhat


def build_complex_expression(u1_val, u2_val, p1_val, p2_val):
    """Build J(u1, u2, p1, p2) = (u1 + u2)^2 * p1 - u1 * p2.
    
    Returns:
        tuple: (Jhat, u1_val, u2_val, p1_val, p2_val)
    """
    u1 = AdjFloat(u1_val)
    u2 = AdjFloat(u2_val)
    p1 = AdjFloat(p1_val)
    p2 = AdjFloat(p2_val)
    sum_u = u1 + u2
    J = sum_u * sum_u * p1 - u1 * p2
    Jhat = ParametrisedReducedFunctional(J, [Control(u1), Control(u2)], [p1, p2])
    return Jhat


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
    Jhat= build_single_control_single_param(c_val, p_val)
    
    # Test initial evaluation
    result = Jhat(c_val)
    expected = c_val * p_val
    assert result == expected
    
    # Test parameter update
    new_p = p_val * mult_factor
    Jhat.parameter_update(new_p)
    result = Jhat(c_val)
    expected = c_val * new_p
    assert result == expected
    
    # Test derivative
    deriv = Jhat.derivative()
    assert deriv[0] == new_p


@pytest.mark.parametrize("c_val,p_val", [
    (2.0, 5.0),
    (1.0, 1.0),
    (3.5, 4.5),
])
def test_parametrised_rf_controls_property(c_val, p_val):
    """Test that controls property returns only user controls, not parameters."""
    Jhat = build_single_control_single_param(c_val, p_val)
    assert len(Jhat.controls) == 1
    assert Jhat.controls[0] is not None

@pytest.mark.parametrize("c_val,p1_val,p2_val,p1_new,p2_new", [
    (2.0, 3.0, 4.0, 5.0, 6.0),
    (1.5, 2.5, 3.5, 4.5, 5.5),
    (3.0, 1.0, 2.0, 3.0, 4.0),
])
def test_parametrised_rf_parameters_property(c_val, p1_val, p2_val, p1_new, p2_new):
    """Test that parameters property returns the current parameter values."""
    Jhat = build_single_control_multi_param(c_val, p1_val, p2_val)
    
    # Check initial parameters
    params = Jhat.parameters
    assert len(params) == 2
    assert params[0] == p1_val
    assert params[1] == p2_val
    
    # Update and check again
    Jhat.parameter_update([p1_new, p2_new])
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
    Jhat = build_multi_control_single_param(c1_val, c2_val, p_val)
    
    # Valid call
    result = Jhat([c1_val, c2_val])
    assert result == c1_val * c2_val * p_val
    
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
def test_parametrised_rf_parameter_update_validation(c_val, p1_val, p2_val):
    """Test that parameter_update validates length of new parameters."""
    Jhat = build_single_control_multi_param(c_val, p1_val, p2_val)
    
    # Valid update
    Jhat.parameter_update([p1_val + 1, p2_val + 1])
    
    # Invalid update - wrong number of parameters
    with pytest.raises(ValueError):
        Jhat.parameter_update([p1_val])  # Only 1 parameter instead of 2
    
    with pytest.raises(ValueError):
        Jhat.parameter_update([p1_val, p2_val, p1_val + 1])  # 3 parameters instead of 2


@pytest.mark.parametrize("c_val,c_new,p_val,p_new", [
    (2.0, 3.0, 5.0, 7.0),
    (1.5, 2.5, 3.0, 4.5),
    (4.0, 1.0, 6.0, 2.0),
])
def test_parametrised_rf_with_single_control_single_parameter(c_val, c_new, p_val, p_new):
    """Test parametrised RF with single control and single parameter at various values."""
    Jhat = build_single_control_single_param(c_val, p_val)
    
    # Test initial evaluation
    result = Jhat(c_new)
    assert result == c_new * p_val
    
    # Test derivative before update
    deriv = Jhat.derivative()
    assert len(deriv) == 1
    assert deriv[0] == p_val
    
    # Update parameter
    Jhat.parameter_update(p_new)
    result = Jhat(c_new)
    assert result == c_new * p_new
    
    # Test derivative after update
    deriv = Jhat.derivative()
    assert len(deriv) == 1
    assert deriv[0] == p_new


@pytest.mark.parametrize("c1_val,c2_val, c1_new, c2_new, p_val,p_new", [
    (2.0, 3.0, 4.0, 6.0, 5.0, 7.0),
    (1.5, 2.5, 3.0, 4.5, 3.0, 4.5),
    (4.0, 1.0, 6.0, 2.0, 2.0, 3.0),
])
def test_parametrised_rf_with_multiple_controls_single_parameter(c1_val, c2_val, c1_new, c2_new, p_val, p_new):
    """Test parametrised RF with multiple controls and single parameter."""
    Jhat = build_multi_control_single_param(c1_val, c2_val, p_val)
    
    # Test initial evaluation
    result = Jhat([c1_new, c2_new])
    assert result == c1_new * c2_new * p_val
    
    # Test initial derivatives
    derivs = Jhat.derivative()
    assert len(derivs) == 2
    assert derivs[0] == c2_new * p_val  # dJ/dc1 = c2 * p
    assert derivs[1] == c1_new * p_val  # dJ/dc2 = c1 * p


    
    # Update parameter
    Jhat.parameter_update(p_new)
    result = Jhat([c1_new, c2_new])
    assert result == c1_new * c2_new * p_new
    
    # Test derivatives after update
    derivs = Jhat.derivative()
    assert derivs[0] == c2_new * p_new  # dJ/dc1 = c2 * p_new
    assert derivs[1] == c1_new * p_new  # dJ/dc2 = c1 * p_new


@pytest.mark.parametrize("c_val, c_new, p1_val,p2_val,p1_new,p2_new", [
    (2.0, 4.0, 3.0, 4.0, 5.0, 6.0),
    (1.5, 2.5, 3.5, 4.5, 5.5, 6.5),
    (3.0, 1.0, 2.0, 3.0, 4.0, 5.0),
])
def test_parametrised_rf_with_single_control_multiple_parameters(c_val, c_new, p1_val, p2_val, p1_new, p2_new):
    """Test parametrised RF with single control and multiple parameters."""
    Jhat = build_single_control_multi_param(c_val, p1_val, p2_val)
    
    # Test initial evaluation
    result = Jhat(c_new)
    assert result == c_new * p1_val * p2_val
    
    # Test initial derivatives
    derivs = Jhat.derivative()
    assert len(derivs) == 1
    assert derivs[0] == p1_val * p2_val  # dJ/dc = p1 * p2
    
    # Update parameters
    Jhat.parameter_update([p1_new, p2_new])
    result = Jhat(c_new)
    assert result == c_new * p1_new * p2_new
    
    # Test derivatives after update
    derivs = Jhat.derivative()
    assert len(derivs) == 1
    assert derivs[0] == p1_new * p2_new  # dJ/dc = p1_new * p2_new


@pytest.mark.parametrize("c1_val,c2_val, c1_new, c2_new, p1_val,p2_val,p1_new,p2_new", [
    (2.0, 3.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0),
    (1.5, 2.5, 3.0, 4.5, 3.0, 4.5, 4.0, 5.5),
    (3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0),
])
def test_parametrised_rf_with_multiple_controls_multiple_parameters(c1_val,c2_val,c1_new,c2_new,p1_val,p2_val,p1_new,p2_new):
    """Test parametrised RF with multiple controls and multiple parameters."""
    Jhat = build_multi_control_multi_param(c1_val, c2_val, p1_val, p2_val)
    
    # Test new evaluation: J = c1^2 * c2 * p1 + c2^2 * p2
    result = Jhat([c1_new, c2_new])
    expected = c1_new * c1_new * c2_new * p1_val + c2_new * c2_new * p2_val
    assert result == expected
    
    # Test newderivatives
    derivs = Jhat.derivative()
    assert len(derivs) == 2
    assert derivs[0] == 2.0 * c1_new * c2_new * p1_val  # dJ/dc1 = 2*c1*c2*p1
    assert derivs[1] == c1_new * c1_new * p1_val + 2.0 * c2_new * p2_val  # dJ/dc2 = c1^2*p1 + 2*c2*p2
    
    # Update parameters
    Jhat.parameter_update([p1_new, p2_new])
    result = Jhat([c1_new, c2_new])
    expected = c1_new * c1_new * c2_new * p1_new + c2_new * c2_new * p2_new
    assert result == expected
    
    # Test derivatives after update
    derivs = Jhat.derivative()
    assert derivs[0] == 2.0 * c1_new * c2_new * p1_new  # dJ/dc1 = 2*c1*c2*p1_new
    assert derivs[1] == c1_new * c1_new * p1_new + 2.0 * c2_new * p2_new  # dJ/dc2 = c1^2*p1_new + 2*c2*p2_new





@pytest.mark.parametrize("c1_val,c2_val,c1_new, c2_new,p1_val,p2_val,p1_new,p2_new", [
    (2.0, 3.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0),
    (1.5, 2.5, 3.0, 4.5, 3.0, 4.5, 4.0, 5.5),
    (3.0, 1.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0),
])
def test_parametrised_rf_complex_expression(c1_val,c2_val,c1_new,c2_new,p1_val,p2_val,p1_new,p2_new):
    """Test parametrised RF with complex mathematical operations J = (c1 + c2)^2 * p1 - c1 * p2."""
    Jhat = build_complex_expression(c1_val, c2_val, p1_val, p2_val)
    
    # Test initial evaluation
    result = Jhat([c1_new, c2_new])
    expected = (c1_new + c2_new) ** 2 * p1_val - c1_new * p2_val
    assert result == expected
    
    # Update and test again
    Jhat.parameter_update([p1_new, p2_new])
    result = Jhat([c1_new, c2_new])
    expected = (c1_new + c2_new) ** 2 * p1_new - c1_new * p2_new
    assert result == expected

@pytest.mark.parametrize("c_val, c_new, p_val, p_new1, p_new2", [
    (2.0, 5.0, 6.0, 7.0, 8.0),
    (1.5, 3.0, 4.0, 5.0, 6.0),
    (4.0, 2.5, 3.0, 4.0, 5.0),
])
def test_parametrised_rf_multiple_parameter_updates(c_val, c_new, p_val, p_new1, p_new2):
    """Test that, in case of multiple parameter updates before a call, the last update is used correctly."""
    Jhat = build_single_control_single_param(c_val, p_val)
    # First update
    Jhat.parameter_update(p_new1)
    # Second update
    Jhat.parameter_update(p_new2)
    # Test evaluation uses the last updated parameter
    result = Jhat(c_new)
    expected = c_new * p_new2
    assert result == expected


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
        Jhat_param_rf = build_single_control_single_param(c_val, p_val)
    
    # Test initial evaluation
    result_rf = Jhat_rf([c_new, p_val])
    result_param_rf = Jhat_param_rf(c_new)
    assert result_rf == result_param_rf
    
    # Update parameter and test again
    result_rf_updated = Jhat_rf([c_new, p_new])
    Jhat_param_rf.parameter_update(p_new)
    result_param_rf_updated = Jhat_param_rf(c_new)
    assert result_rf_updated == result_param_rf_updated

    # Test derivatives
    derivs_rf = Jhat_rf.derivative()
    derivs_param_rf = Jhat_param_rf.derivative()
    assert derivs_rf[0] == derivs_param_rf[0]  # dJ/dc should be the same
