from enum import Enum
from itertools import pairwise


class CheckpointError(RuntimeError):
    pass


class Mode(Enum):
    RECORD = 1
    EVALUATED = 2
    EXHAUSTED = 3


class CheckpointManager:
    def __init__(self, schedule, tape):
        self.schedule = schedule
        self.tape = tape
        self.timesteps = schedule.function.l
        self.mode = Mode.RECORD
        self._iterator = iter(self.schedule)
        self._current = next(self._iterator)
        self._checkpointable_state = set()
        # Tell the tape to only checkpoint input data until told otherwise.
        self.tape.latest_checkpoint = 0
        # Process any initial instructions on the tape.
        self.end_timestep(-1)

    def end_timestep(self, timestep):
        """Process the end of a timestep while recording."""
        if self.mode != Mode.RECORD:
            raise CheckpointError(f"Cannot end timestep in {self.mode}")

        while not self.process_taping(self._current, timestep):
            self._current = next(self._iterator)

    def end_taping(self):
        current_timestep = self.tape.latest_timestep
        while self.mode != Mode.Evaluated:
            self.end_timestep(current_timestep)
            current_timestep += 1

    def process_taping(self, operation, timestep):
        """Perform any actions demanded by operation at timestep.

        If execution should now continue, return True. Otherwise return
        False and the next operation in the schedule will be processed.
        """
        if operation.type == "Forwards":
            try:
                if timestep < (operation.index[0] - 1):
                    raise CheckpointError("Timestep is before start of Forward operation.")
                return timestep < operation.index[1]
            except IndexError:
                return timestep <= operation.index
        elif operation.type == "Write_memory":
            self.tape.latest_checkpoint = operation.index
            return False
        elif operation.type == "Backward":
            # We're specifically allowed to encounter the backward at the end of the tape.
            if timestep == operation.index - 1:
                self.mode = Mode.EVALUATED
                return True
            else:
                raise CheckpointError("Backward encountered while taping.")
        raise CheckpointError(f"Unable to process {operation.type} while taping.")

    def evaluate_adj(self, last_block, markings):
        assert last_block == 0  # Work out other cases when they arise.
        if self.mode == Mode.RECORD:
            # The declared timesteps were not exhausted while taping.
            self.end_taping()
        if self.mode != Mode.EVALUATED:
            raise CheckpointError("Evaluate Functional before calling gradient.")

        # The first backward on the tape will already be current.
        assert self._current.type == "Backward"
        self.process_operation(self._current, markings)
        for operation in self._iterator:
            self.process_operation(operation, markings)

        self.mode = Mode.EXHAUSTED

    def process_operation(self, operation, markings):
        """Perform the operations required by the schedule."""
        if operation.type == "Forwards":
            # Advance, only keeping checkpointable state.
            try:
                timesteps = (operation.index[0], operation.index[1] + 1)
            except IndexError:
                timesteps = (operation.index, operation.index + 2)
            for step, next_step in pairwise(
                self.tape.timesteps[slice(*timesteps)]
            ):
                for block in step:
                    block.recompute()
                for block in step:
                    for var in block.get_outputs():
                        if var not in next_step.checkpointable_state:
                            var._checkpoint = None
                    for var in self._checkpointable_state - next_step.checkpointable_state:
                        var._checkpoint = None
                    self._checkpointable_state = next_step.checkpointable_state
        elif operation.type == "Backward":
            # Evaluate this step storing all outputs, then evaluate the adjoint.
            step = self.tape.timesteps[operation.index]
            for block in step:
                block.recompute()
            for block in reversed(step):
                block.evaluate_adj(markings=markings)
            # Output variables are used for the last time when running backwards.
            for block in step:
                for var in block.get_outputs():
                    var._checkpoint = None
                    var.reset_variables(("tlm",))
                    if not var.is_control:
                        var.reset_variables(("adjoint", "hessian"))
        elif operation.type == "Write_memory":
            # Clear the current checkpointable state so it isn't deleted by the
            # next forward.
            self._checkpointable_state = set()
        elif operation.type == "Read_memory":
            # We're rewinding to a saved checkpoint, so clear the current
            # checkpointable state.
            self._checkpointable_state = set()
        elif operation.type == "Discard_memory":
            if operation.index != 0:
                step = self.tape.timesteps[operation.index]
                for var in step.checkpointable_state:
                    var._checkpoint = None
