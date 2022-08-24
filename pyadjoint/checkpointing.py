from enum import Enum
from functools import singledispatchmethod
from tlm_adjoint.checkpoint_schedules import Clear, \
    Configure, EndForward, EndReverse, Forward, Reverse, Read, Write


class CheckpointError(RuntimeError):
    pass


class Mode(Enum):
    RECORD = 1
    FINISHED_RECORDING = 2
    EVALUATED = 3
    EXHAUSTED = 4
    RECOMPUTE = 5
    EVALUATE_ADJOINT = 6


class AdjointSchedule(list):
    def __init__(self, operations=(), steps=0):
        # Steps is only intended to provide a rough indicator for progress
        # bars.
        super().__init__(operations)
        self.steps = steps


def process_schedule(schedule):
    schedule = list(schedule)

    end_forward = schedule.index(EndForward()) + 1

    forward_steps = sum(map(len, schedule[:end_forward]))
    reverse_steps = sum(map(len, schedule[end_forward:]))

    forward = AdjointSchedule(schedule[:end_forward], forward_steps)
    reverse = AdjointSchedule(schedule[end_forward:], reverse_steps)

    return forward, reverse


class CheckpointManager:
    def __init__(self, schedule, tape):
        self.forward, self.reverse = process_schedule(schedule)
        if schedule.uses_disk_storage and not tape._package_data:
            raise CheckpointError(
                "The schedule employs disk checkpointing but it is not configured."
            )
        self.tape = tape
        self.timesteps = schedule.max_n()
        self.mode = Mode.RECORD
        self._iterator = iter(self.forward)
        self._current = next(self._iterator)
        self._configuration = None
        self._configuration_step = None
        self._stored_timesteps = []
        self._checkpointable_state = set()
        # Tell the tape to only checkpoint input data until told otherwise.
        self.tape.latest_checkpoint = 0
        # Process any initial instructions on the tape.
        self.end_timestep(-1)

    def end_timestep(self, timestep):
        """Process the end of a timestep while recording."""
        if self.mode == Mode.EVALUATED:
            raise CheckpointError("Not enough timesteps in schedule.")
        elif self.mode != Mode.RECORD:
            raise CheckpointError(f"Cannot end timestep in {self.mode}")

        while not self.process_taping(self._current, timestep + 1):
            self._current = next(self._iterator)

    def end_taping(self):
        current_timestep = self.tape.latest_timestep
        while self.mode != Mode.EVALUATED:
            self.end_timestep(current_timestep)
            current_timestep += 1

    @singledispatchmethod
    def process_taping(self, operation, timestep):
        """Perform any actions demanded by operation at timestep.

        If execution should now continue, return True. Otherwise return
        False and the next operation in the schedule will be processed.
        """
        raise CheckpointError(f"Unable to process {operation} while taping.")

    @process_taping.register(Forward)
    def _(self, operation, timestep):

        if timestep < (operation.n0):
            raise CheckpointError(
                "Timestep is before start of Forward operation."
            )
        if timestep in operation:
            self.tape.get_blocks().append_step()
            return True
        else:
            return False

    @process_taping.register(Clear)
    def _(self, operation, timestep):
        if not self._configuration:
            return False

        if operation.clear_data:
            for step in range(self._configuration_step, timestep):
                for block in self.tape.timesteps[step]:
                    for output in block.get_outputs():
                        output.checkpoint = None
        if operation.clear_ics:
            for var in self.tape.timesteps[self._configuration_step]:
                var.checkpoint = None

        return False

    @process_taping.register(Configure)
    def _(self, operation, timestep):
        if operation.store_ics:
            self.tape.latest_checkpoint = timestep
        self._configuration = operation
        self._configuration_step = timestep

        self.tape._eagerly_checkpoint_outputs = operation.store_data

        return False

    @process_taping.register(EndForward)
    def _(self, operation, timestep):
        self.mode = Mode.EVALUATED
        return True

    def recompute(self, functional=None):
        self.mode = Mode.RECOMPUTE

        with self.tape.progress_bar("Evaluating Functional",
                                    max=self.forward.steps) as bar:
            for operation in self.forward:
                self.process_operation(operation, bar, functional=functional)

    def evaluate_adj(self, last_block, markings):
        assert last_block == 0  # Work out other cases when they arise.
        if self.mode == Mode.RECORD:
            # The declared timesteps were not exhausted while taping.
            self.end_taping()
        if self.mode not in (Mode.EVALUATED, Mode.FINISHED_RECORDING):
            raise CheckpointError("Evaluate Functional before calling gradient.")

        with self.tape.progress_bar("Evaluating Adjoint",
                                    max=self.reverse.steps) as bar:
            for operation in self.reverse:
                self.process_operation(operation, bar, markings=markings)
                # Only set the mode after the first backward in order to handle
                # that step correctly.
                self.mode = Mode.EVALUATE_ADJOINT

    @singledispatchmethod
    def process_operation(self, operation, bar, **kwargs):
        """Perform the operations required by the schedule."""
        raise CheckpointError(f"Unable to process {operation}.")

    @process_taping.register(Write)
    @process_operation.register(Write)
    def _(self, operation, bar, **kwargs):
        for data in self.tape._package_data.values():
            data.configure_checkpointing(operation.storage)

        self.tape.timesteps[operation.n].checkpoint()

        for data in self.tape._package_data.values():
            data.configure_checkpointing("RAM")

    @process_operation.register(Forward)
    def _(self, operation, bar, functional=None, **kwargs):
        for step in operation:
            bar.next()
            current_step = self.tape.timesteps[step]
            self._stored_timesteps.append(step)
            for block in current_step:
                block.recompute()
            if not self._configuration.store_data:
                next_step = self.tape.timesteps[step + 1]
                to_keep = next_step.checkpointable_state
                if functional:
                    to_keep = to_keep.union([functional.block_variable])
                for block in current_step:
                    for var in block.get_outputs():
                        if var not in to_keep:
                            var.checkpoint = None
                    for var in self._checkpointable_state - to_keep:
                        var.checkpoint = None
                    self._checkpointable_state = next_step.checkpointable_state

    @process_operation.register(Reverse)
    def _(self, operation, bar, markings, **kwargs):
        for step in operation:
            bar.next()
            current_step = self.tape.timesteps[step]
            for block in reversed(current_step):
                block.evaluate_adj(markings=markings)
            # Output variables are used for the last time when running backwards.
            for block in current_step:
                for var in block.get_outputs():
                    var.checkpoint = None
                    var.reset_variables(("tlm",))
                    if not var.is_control:
                        var.reset_variables(("adjoint", "hessian"))
        # Not currently advancing so no checkpointable state.
        self._checkpointable_state = set()

    @process_operation.register(Read)
    def _(self, operation, bar, **kwargs):
        current_step = self.tape.timesteps[operation.n]
        current_step.restore_from_checkpoint()
        self._checkpointable_state = current_step.checkpointable_state
        if operation.delete:
            current_step.delete_checkpoint()

    @process_operation.register(EndForward)
    def _(self, operation, bar, **kwargs):
        self.mode = Mode.EVALUATED

    @process_operation.register(EndReverse)
    def _(self, operation, bar, **kwargs):
        if operation.exhausted:
            self.mode = Mode.EXHAUSTED
        else:
            self.mode = Mode.EVALUATED

    @process_operation.register(Configure)
    def _(self, operation, bar, **kwargs):
        if operation.store_ics:
            # Clear the current checkpointable state so it isn't deleted by the
            # next forward.
            self._checkpointable_state = set()
        self._configuration = operation
        self._configuration_step = None

    @process_operation.register(Clear)
    def _(self, operation, bar, functional=None, **kwargs):
        # Return later to what we do if we're not clearing all of these.
        assert operation.clear_ics
        assert operation.clear_data

        to_keep = self._checkpointable_state
        if functional:
            to_keep = to_keep.union([functional.block_variable])
        for step in self._stored_timesteps:
            for block in self.tape.timesteps[step]:
                for output in block.get_outputs():
                    if output not in to_keep:
                        output.checkpoint = None
