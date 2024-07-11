from contextlib import contextmanager

import pytest

from pyadjoint import *


@pytest.fixture(autouse=True)
def _():
    get_working_tape().clear_tape()
    continue_annotation()
    continue_reverse_over_forward()
    yield
    get_working_tape().clear_tape()
    pause_annotation()
    pause_reverse_over_forward()


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
def test_exp(a_val, tlm_a_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = exp(a)
    _ = compute_gradient(b.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert adj_value == exp(a_val) * tlm_a_val


@pytest.mark.parametrize("a_val", [2.0, 3.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
def test_log(a_val, tlm_a_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = log(a)
    _ = compute_gradient(b.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert adj_value == -tlm_a_val / (a_val ** 2)
