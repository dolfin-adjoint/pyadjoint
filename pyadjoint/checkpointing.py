"""Utilities to support disk checkpointing of adjoint simulations."""
from .block import Block
from .tape import get_working_tape


def adjoint_step():
    """Symbolically start a new step of the computation.

    This will usually be a new timestep, but it can be any arbitrary unit of
    computation that the user wishes to use as a basis for checkpoints."""

    tape = get_working_tape()

    tape.add_block(StepBlock(tape._last_step))
    tape._last_step += 1


class StepBlock(Block):
    """A Block to symbolically record the start of a step on the tape.

    A :class:`StepBlock` is inserted onto the tape every time the user calls
    :func:`adjoint_step`. It has no result and initially has no dependencies.
    The tape is then analysed and dependencies are added for every
    :class:`BlockVariable` which is used after this point but defined before
    this point."""
    def __init__(self, step_number):
        super().__init__()
        self.step_number = step_number

    def __str__(self):
        return f"Step {self.step_number}"
