from contextlib import contextmanager

import numpy as np
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
    x = exp(a)
    stop_annotating()
    _ = compute_gradient(x.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert np.allclose(adj_value, exp(a_val) * tlm_a_val)


@pytest.mark.parametrize("a_val", [2.0, 3.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
def test_log(a_val, tlm_a_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    x = log(a)
    stop_annotating()
    _ = compute_gradient(x.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert np.allclose(adj_value, -tlm_a_val / (a_val ** 2))


@pytest.mark.parametrize("a_val", [2.0, 3.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
@pytest.mark.parametrize("c", [0, 1])
def test_min_left(a_val, tlm_a_val, c):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(a_val + c)
    x = min(a, b)
    assert x.block_variable.tlm_value == tlm_a_val
    y = x ** 3
    stop_annotating()
    _ = compute_gradient(y.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert np.allclose(adj_value, 6 * a_val * tlm_a_val)



@pytest.mark.parametrize("b_val", [2.0, 3.0])
@pytest.mark.parametrize("tlm_b_val", [3.5, -3.5])
def test_min_right(b_val, tlm_b_val):
    a = AdjFloat(b_val + 1)
    b = AdjFloat(b_val)
    b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = min(a, b)
    assert x.block_variable.tlm_value == tlm_b_val
    y = x ** 3
    stop_annotating()
    _ = compute_gradient(y.block_variable.tlm_value, Control(b))
    adj_value = b.block_variable.adj_value
    assert np.allclose(adj_value, 6 * b_val * tlm_b_val)


@pytest.mark.parametrize("a_val", [2.0, 3.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
@pytest.mark.parametrize("c", [0, -1])
def test_max_left(a_val, tlm_a_val, c):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(a_val + c)
    x = max(a, b)
    assert x.block_variable.tlm_value == tlm_a_val
    y = x ** 3
    stop_annotating()
    _ = compute_gradient(y.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert np.allclose(adj_value, 6 * a_val * tlm_a_val)


@pytest.mark.parametrize("b_val", [2.0, 3.0])
@pytest.mark.parametrize("tlm_b_val", [3.5, -3.5])
def test_max_right(b_val, tlm_b_val):
    a = AdjFloat(b_val - 1)
    b = AdjFloat(b_val)
    b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = max(a, b)
    assert x.block_variable.tlm_value == tlm_b_val
    y = x ** 3
    stop_annotating()
    _ = compute_gradient(y.block_variable.tlm_value, Control(b))
    adj_value = b.block_variable.adj_value
    assert np.allclose(adj_value, 6 * b_val * tlm_b_val)


@pytest.mark.parametrize("a_val", [2.0, 3.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5, None])
@pytest.mark.parametrize("b_val", [4.25, 5.25])
@pytest.mark.parametrize("tlm_b_val", [5.8125, -5.8125, None])
def test_pow(a_val, tlm_a_val, b_val, tlm_b_val):
    a = AdjFloat(a_val)
    if tlm_a_val is not None:
        a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(b_val)
    if tlm_b_val is not None:
        b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = a ** b
    if tlm_a_val is not None or tlm_b_val is not None:
        _ = compute_gradient(x.block_variable.tlm_value, (Control(a), Control(b)))
        assert np.allclose(
            a.block_variable.adj_value,
            b_val * (b_val - 1) * (a_val ** (b_val - 2)) * (0.0 if tlm_a_val is None else tlm_a_val)
            + (1 + b_val * log(a_val)) * (a_val ** (b_val - 1)) * (0.0 if tlm_b_val is None else tlm_b_val))
        assert np.allclose(
            b.block_variable.adj_value,
            (1 + b_val * log(a_val)) * (a_val ** (b_val - 1)) * (0.0 if tlm_a_val is None else tlm_a_val)
            + (log(a_val) ** 2) * (a_val ** b_val) * (0.0 if tlm_b_val is None else tlm_b_val))
        

@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5, None])
@pytest.mark.parametrize("b_val", [4.25, -4.25])
@pytest.mark.parametrize("tlm_b_val", [5.8125, -5.8125, None])
def test_add(a_val, tlm_a_val, b_val, tlm_b_val):
    a = AdjFloat(a_val)
    if tlm_a_val is not None:
        a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(b_val)
    if tlm_b_val is not None:
        b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = a + b
    if tlm_a_val is None and tlm_b_val is None:
        assert x.block_variable.tlm_value is None
    else:
        assert (x.block_variable.tlm_value ==
                (0.0 if tlm_a_val is None else tlm_a_val)
                + (0.0 if tlm_b_val is None else tlm_b_val))
    y = x ** 3
    stop_annotating()
    if tlm_a_val is not None or tlm_b_val is not None:
        _ = compute_gradient(y.block_variable.tlm_value, (Control(a), Control(b)))
        assert np.allclose(
            a.block_variable.adj_value,
            (6 * a_val + 6 * b_val) * (0.0 if tlm_a_val is None else tlm_a_val)
            + (6 * a_val + 6 * b_val) * (0.0 if tlm_b_val is None else tlm_b_val))
        assert np.allclose(
            b.block_variable.adj_value,
            (6 * a_val + 6 * b_val) * (0.0 if tlm_a_val is None else tlm_a_val)
            + (6 * a_val + 6 * b_val) * (0.0 if tlm_b_val is None else tlm_b_val))


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5, None])
@pytest.mark.parametrize("b_val", [4.25, -4.25])
@pytest.mark.parametrize("tlm_b_val", [5.8125, -5.8125, None])
def test_sub(a_val, tlm_a_val, b_val, tlm_b_val):
    a = AdjFloat(a_val)
    if tlm_a_val is not None:
        a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(b_val)
    if tlm_b_val is not None:
        b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = a - b
    if tlm_a_val is None and tlm_b_val is None:
        assert x.block_variable.tlm_value is None
    else:
        assert (x.block_variable.tlm_value ==
                (0.0 if tlm_a_val is None else tlm_a_val)
                - (0.0 if tlm_b_val is None else tlm_b_val))
    y = x ** 3
    stop_annotating()
    if tlm_a_val is not None or tlm_b_val is not None:
        _ = compute_gradient(y.block_variable.tlm_value, (Control(a), Control(b)))
        assert np.allclose(
            a.block_variable.adj_value,
            (6 * a_val - 6 * b_val) * (0.0 if tlm_a_val is None else tlm_a_val)
            + (-6 * a_val + 6 * b_val) * (0.0 if tlm_b_val is None else tlm_b_val))
        assert np.allclose(
            b.block_variable.adj_value,
            (-6 * a_val + 6 * b_val) * (0.0 if tlm_a_val is None else tlm_a_val)
            + (6 * a_val - 6 * b_val) * (0.0 if tlm_b_val is None else tlm_b_val))


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5, None])
@pytest.mark.parametrize("b_val", [4.25, -4.25])
@pytest.mark.parametrize("tlm_b_val", [5.8125, -5.8125, None])
def test_mul(a_val, tlm_a_val, b_val, tlm_b_val):
    a = AdjFloat(a_val)
    if tlm_a_val is not None:
        a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(b_val)
    if tlm_b_val is not None:
        b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = a * b
    if tlm_a_val is None and tlm_b_val is None:
        assert x.block_variable.tlm_value is None
    else:
        assert (x.block_variable.tlm_value ==
                b_val * (0.0 if tlm_a_val is None else tlm_a_val)
                + a_val * (0.0 if tlm_b_val is None else tlm_b_val))
    stop_annotating()
    if tlm_a_val is not None or tlm_b_val is not None:
        _ = compute_gradient(x.block_variable.tlm_value, (Control(a), Control(b)))
        if tlm_b_val is None:
            assert a.block_variable.adj_value is None
        else:
            assert np.allclose(
                a.block_variable.adj_value, tlm_b_val)
        if tlm_a_val is None:
            assert b.block_variable.adj_value is None
        else:
            assert np.allclose(
                b.block_variable.adj_value, tlm_a_val)


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
@pytest.mark.parametrize("b_val", [4.25, -4.25])
@pytest.mark.parametrize("tlm_b_val", [5.8125, -5.8125])
def test_div(a_val, tlm_a_val, b_val, tlm_b_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    b = AdjFloat(b_val)
    b.block_variable.tlm_value = AdjFloat(tlm_b_val)
    x = (a ** 2) / b
    stop_annotating()
    _ = compute_gradient(x.block_variable.tlm_value, (Control(a), Control(b)))
    assert np.allclose(
        a.block_variable.adj_value,
        (2 / b_val) * tlm_a_val - 2 * a_val / (b_val ** 2) * tlm_b_val)
    assert np.allclose(
        b.block_variable.adj_value,
        - 2 * a_val / (b_val ** 2) * tlm_a_val + 2 * (a_val ** 2) / (b_val ** 3) * tlm_b_val)


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
def test_pos(a_val, tlm_a_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    x = +a
    assert x.block_variable.tlm_value == tlm_a_val
    y = x ** 3
    stop_annotating()
    _ = compute_gradient(y.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert np.allclose(adj_value, 6 * a_val * tlm_a_val)


@pytest.mark.parametrize("a_val", [2.0, -2.0])
@pytest.mark.parametrize("tlm_a_val", [3.5, -3.5])
def test_neg(a_val, tlm_a_val):
    a = AdjFloat(a_val)
    a.block_variable.tlm_value = AdjFloat(tlm_a_val)
    x = -a
    assert x.block_variable.tlm_value == -tlm_a_val
    y = x ** 3
    stop_annotating()
    _ = compute_gradient(y.block_variable.tlm_value, Control(a))
    adj_value = a.block_variable.adj_value
    assert np.allclose(adj_value, -6 * a_val * tlm_a_val)
