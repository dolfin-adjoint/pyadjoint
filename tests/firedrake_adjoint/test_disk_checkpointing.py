import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.adjoint import *
import numpy as np
import os


def adjoint_example(fine, coarse):
    dg_space = FunctionSpace(fine, "DG", 1)
    cg_space = FunctionSpace(fine, "CG", 2)
    W = dg_space * cg_space

    w = Function(W)

    x, y = SpatialCoordinate(fine)
    # InterpolateBlock
    m = interpolate(sin(4*pi*x)*cos(4*pi*y), cg_space)

    u, v = w.split()
    # FunctionAssignBlock, FunctionMergeBlock
    v.assign(m)
    # FunctionSplitBlock, GenericSolveBlock
    u.project(v)

    dg_space_c = FunctionSpace(coarse, "DG", 1)
    cg_space_c = FunctionSpace(coarse, "CG", 2)

    # SupermeshProjectBlock
    u_c = project(u, dg_space_c)
    v_c = project(v, cg_space_c)

    # AssembleBlock
    J = assemble((u_c - v_c)**2 * dx)

    Jhat = ReducedFunctional(J, Control(m))

    with stop_annotating():
        m_new = interpolate(sin(4*pi*x)*cos(4*pi*y), cg_space)
    checkpointer = get_working_tape()._package_data["firedrake"]
    init_file_timestamp = os.stat(checkpointer.init_checkpoint_file.name).st_mtime
    current_file_timestamp = os.stat(checkpointer.current_checkpoint_file.name).st_mtime
    Jnew = Jhat(m_new)
    # Check that any new disk checkpoints are written to the correct file.
    assert init_file_timestamp == os.stat(checkpointer.init_checkpoint_file.name).st_mtime
    assert current_file_timestamp < os.stat(checkpointer.current_checkpoint_file.name).st_mtime

    assert np.allclose(J, Jnew)

    grad_Jnew = Jhat.derivative()

    return Jnew, grad_Jnew


def test_disk_checkpointing():
    # Use a Firedrake Tape subclass that supports disk checkpointing.
    set_working_tape(Tape())
    tape = get_working_tape()
    tape.clear_tape()
    enable_disk_checkpointing()

    fine = checkpointable_mesh(UnitSquareMesh(10, 10, name="fine"))
    coarse = checkpointable_mesh(UnitSquareMesh(4, 4, name="coarse"))
    J_disk, grad_J_disk = adjoint_example(fine, coarse)

    tape.clear_tape()
    pause_disk_checkpointing()

    J_mem, grad_J_mem = adjoint_example(fine, coarse)

    assert np.allclose(J_disk, J_mem)
    assert np.allclose(assemble((grad_J_disk - grad_J_mem)**2*dx), 0.0)
    tape.clear_tape()


if __name__ == "__main__":
    test_disk_checkpointing()
