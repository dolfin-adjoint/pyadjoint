import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import Revolve, MultistageCheckpointSchedule
import numpy as np
from collections import deque


def spring_mass_damper(displacement_0, total_steps, space):
    continue_annotation()
    dt = 0.01
    stiff = Constant(2.0)
    damping = Constant(0.1)
    rho = Constant(0.1)
    # Adams-Bashforth coefficients.
    adams_bashforth_coeffs = [55.0/24.0, -59.0/24.0, 37.0/24.0, -3.0/8.0]
    # Adams-Moulton coefficients.
    adams_moulton_coeffs = [9.0/24.0, 19.0/24.0, -5.0/24.0, 1.0/24.0]
    M = len(adams_bashforth_coeffs)
    displacement = Function(space)
    velocity = deque([Function(space) for m in range(M)])
    F = deque([Function(space) for m in range(M)])
    displacement.assign(displacement_0)
    
    for _ in get_working_tape().timestepper(iter(range(total_steps))):
        for _ in range(M - 1):
            F.append(F.popleft())
        F[0].assign(- (stiff * displacement + damping * velocity[0])/rho)
        for _ in range(M - 1):
            velocity.append(velocity.popleft())
        for m in range(M):
            velocity[0].assign(velocity[0] + dt * adams_bashforth_coeffs[m] * F[m])
        for m in range(M):
            displacement.assign(displacement + dt * adams_moulton_coeffs[m] * velocity[m])
    return assemble(displacement * displacement * dx)


@pytest.mark.parametrize("checkpointing",
                         [("Revolve"),
                          ("Multistage"),
                          (None),
                          ])
def test_multisteps(checkpointing):
    total_steps = 100
    mesh = UnitIntervalMesh(1)
    space = FunctionSpace(mesh, "DG", 0)
    tape = get_working_tape()
    tape.progress_bar = ProgressBar
    if checkpointing == "Revolve":
        tape.enable_checkpointing(Revolve(total_steps, total_steps//3))
    if checkpointing == "Multistage":
        tape.enable_checkpointing(MultistageCheckpointSchedule(total_steps, total_steps//3, 0))
    displacement_0 = Function(space).assign(1.0)
    J = spring_mass_damper(displacement_0, total_steps, space)
    c = Control(displacement_0)
    J_hat = ReducedFunctional(J, c)
    dJ = J_hat.derivative()
    # Recomputing the functional with a modified control variable
    # before the recompute test.
    J_hat(Function(space).assign(0.5))
    # Recompute test
    assert(np.allclose(J_hat(displacement_0), J))

    dJbar = J_hat.derivative()
    # Test recompute adjoint-based gradient
    assert np.allclose(dJ.dat.data_ro[:], dJbar.dat.data_ro[:])
    h = Function(space)
    h.assign(1, annotate=False)
    taylor_test(J_hat, displacement_0, h)