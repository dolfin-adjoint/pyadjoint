"""Utilities to support disk checkpointing of adjoint simulations."""
from .block import Block
from .tape import get_working_tape


def adjoint_step():
    """Symbolically start a new step of the computation.

    This will usually be a new timestep, but it can be any arbitrary unit of
    computation that the user wishes to use as a basis for checkpoints."""

    tape = get_working_tape()

    idx = tape.add_block(StepBlock(len(tape.steps)))
    tape.steps.append(idx)


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


class CheckpointManager:
    """An object controlling the execution of a Tape using checkpointing."""
    def __init__(self, tape):
        self.tape = tape
        self.state = None
        if not self.tape.steps:
            self.tape.find_steps()
        if not self.tape.steps:
            raise ValueError(
                "You must specify adjoint_steps in order to use checkpointing."
            )
        self.next_step = 0
    
    def advance(self, finish, store=False):
        pass

    def take_shot(self):
        pass

    def erase(self, step):
        """Remove from memory the stored dependencies at the end of step."""
        pass

    def adjoint(self):
        # Compute the adjoint to the last step.
        pass



