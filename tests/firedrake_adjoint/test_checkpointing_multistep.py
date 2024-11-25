import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import MixedCheckpointSchedule, StorageType
import numpy as np
from collections import deque
continue_annotation()
total_steps = 20
dt = 0.01
mesh = UnitIntervalMesh(1)
V = FunctionSpace(mesh, "DG", 0)


def _check_forward(tape):
    for current_step in tape.timesteps[1:-1]:
        for block in current_step:
            for deps in block.get_dependencies():
                if (
                    deps not in tape.timesteps[0].checkpointable_state
                    and deps not in tape.timesteps[-1].checkpointable_state
                ):
                    assert deps._checkpoint is None
            for out in block.get_outputs():
                if out not in tape.timesteps[-1].checkpointable_state:
                    assert out._checkpoint is None


def _check_recompute(tape):
    for current_step in tape.timesteps[1:-1]:
        for block in current_step:
            for deps in block.get_dependencies():
                if deps not in tape.timesteps[0].checkpointable_state:
                    assert deps._checkpoint is None
            for out in block.get_outputs():
                assert out._checkpoint is None

    for block in tape.timesteps[0]:
        for out in block.get_outputs():
            assert out._checkpoint is None
    for block in tape.timesteps[len(tape.timesteps)-1]:
        for deps in block.get_dependencies():
            if (
                deps not in tape.timesteps[0].checkpointable_state
                and deps not in tape.timesteps[len(tape.timesteps)-1].adjoint_dependencies
            ):
                assert deps._checkpoint is None


def _check_reverse(tape):
    for step, current_step in enumerate(tape.timesteps):
        if step > 0:
            for block in current_step:
                for deps in block.get_dependencies():
                    if deps not in tape.timesteps[0].checkpointable_state:
                        assert deps._checkpoint is None
                for out in block.get_outputs():
                    assert out._checkpoint is None
        elif step == 0:
            for block in current_step:
                for out in block.get_outputs():
                    assert out._checkpoint is None


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
    tape.enable_checkpointing(MixedCheckpointSchedule(total_steps, 2, storage=StorageType.RAM))
    displacement_0 = Function(V).assign(1.0)
    val = J(displacement_0)
    _check_forward(tape)
    c = Control(displacement_0)
    J_hat = ReducedFunctional(val, c)
    dJ = J_hat.derivative()
    _check_reverse(tape)
    # Recomputing the functional with a modified control variable
    # before the recompute test.
    J_hat(Function(V).assign(0.5))
    _check_recompute(tape)
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
    tape.enable_checkpointing(MixedCheckpointSchedule(total_steps, 2, storage=StorageType.RAM))
    val = J(displacement_0)
    J_hat = ReducedFunctional(val, Control(displacement_0))
    dJ = J_hat.derivative()
    val_recomputed = J_hat(displacement_0)
    assert np.allclose(val_recomputed, val_recomputed0)
    assert np.allclose(dJ.dat.data_ro[:], dJ0.dat.data_ro[:])
