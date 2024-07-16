import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import Revolve
import numpy as np
from collections import deque
continue_annotation()
total_steps = 20
dt = 0.01
mesh = UnitIntervalMesh(1)
V = FunctionSpace(mesh, "DG", 0)


def J(displacement_0):
    stiff = Constant(2.5)
    damping = Constant(0.3)
    rho = Constant(1.0)
    # Adams-Bashforth coefficients.
    adams_bashforth_coeffs = [55.0/24.0, -59.0/24.0, 37.0/24.0, -3.0/8.0]
    # Adams-Moulton coefficients.
    adams_moulton_coeffs = [9.0/24.0, 19.0/24.0, -5.0/24.0, 1.0/24.0]
    displacement = Function(V)
    velocity = deque([Function(V) for _ in adams_bashforth_coeffs])
    forcing = deque([Function(V) for _ in adams_bashforth_coeffs])
    displacement.assign(displacement_0)
    tape = get_working_tape()
    for _ in tape.timestepper(range(total_steps)):
        for _ in range(len(adams_bashforth_coeffs) - 1):
            forcing.append(forcing.popleft())
        forcing[0].assign(-(stiff * displacement + damping * velocity[0])/rho)
        for _ in range(len(adams_bashforth_coeffs) - 1):
            velocity.append(velocity.popleft())
        for m, coef in enumerate(adams_bashforth_coeffs):
            velocity[0].assign(velocity[0] + dt * coef * forcing[m])
        for m, coef in enumerate(adams_moulton_coeffs):
            displacement.assign(displacement + dt * coef * velocity[m])
    return assemble(displacement * displacement * dx)


def test_multisteps():
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    tape.enable_checkpointing(Revolve(total_steps, 2))
    displacement_0 = Function(V).assign(1.0)
    val = J(displacement_0)
    c = Control(displacement_0)
    J_hat = ReducedFunctional(val, c)
    dJ = J_hat.derivative()
    # Recomputing the functional with a modified control variable
    # before the recompute test.
    J_hat(Function(V).assign(0.5))
    # Recompute test
    assert(np.allclose(J_hat(displacement_0), val))
    # Test recompute adjoint-based gradient
    assert np.allclose(dJ.dat.data_ro[:], J_hat.derivative().dat.data_ro[:])
    assert taylor_test(J_hat, displacement_0, Function(V).assign(1, annotate=False)) > 1.9


def test_validity():
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    displacement_0 = Function(V).assign(1.0)
    # Without checkpointing.
    val0 = J(displacement_0)
    J_hat0 = ReducedFunctional(val0, Control(displacement_0))
    dJ0 = J_hat0.derivative()
    val_recomputed0 = J(displacement_0)
    tape.clear_tape()

    # With checkpointing.
    tape.enable_checkpointing(Revolve(total_steps, 2))
    val = J(displacement_0)
    J_hat = ReducedFunctional(val, Control(displacement_0))
    dJ = J_hat.derivative()
    val_recomputed = J_hat(displacement_0)
    assert np.allclose(val_recomputed, val_recomputed0)
    assert np.allclose(dJ.dat.data_ro[:], dJ0.dat.data_ro[:])


def test_checkpointing_delegate_cofunction():
    tape = get_working_tape()
    n_steps = 3
    tape.enable_checkpointing(Revolve(n_steps, n_steps))

    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "R", 0)
    v = TestFunction(V)
    c = Constant(1.0)
    b = c * v * dx
    u1 = Cofunction(V.dual(), name="u1")
    sol = Function(V, name="sol")
    u = TrialFunction(V)
    u0 = assemble(b)
    J = 0
    for i in tape.timestepper(iter(range(n_steps))):
        u1.assign(Constant(i) * u0)
        solve(u * v * dx == u1, sol)
        J += assemble(sol * sol * dx)
    J_hat = ReducedFunctional(J, Control(c))
    assert np.isclose(J_hat(c), J)
