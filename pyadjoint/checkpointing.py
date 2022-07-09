from enum import Enum


class CheckpointError(RuntimeError):
    pass


class Mode(Enum):
    RECORD = 1


class CheckpointManager:
    def __init__(self, schedule, tape):
        self.schedule = schedule
        self.tape = tape
        self.timesteps = schedule.function.l
        self.mode = Mode.RECORD
        self._iterator = None
        self._current = None
        # Tell the tape to only checkpoint input data until told otherwise.
        tape.latest_checkpoint = 0

    def end_timestep(self, timestep):
        """Process the end of a timestep while recording."""
        if self.mode != Mode.RECORD:
            raise CheckpointError(f"Cannot end timestep in {self.mode}")
        if not self._iterator:
            self._iterator = iter(self.schedule)
            self._current = next(self._iterator)

        while not self.process_taping(self._current, timestep):
            self._current = next(self._iterator)

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
                return True
            else:
                raise CheckpointError("Backward encountered while taping.")
        raise CheckpointError(f"Unable to process {operation.type} while taping.")
