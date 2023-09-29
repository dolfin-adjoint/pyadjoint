from enum import Enum
from functools import singledispatchmethod
from checkpoint_schedules import (Copy, Move, EndForward, EndReverse, Forward, Reverse, StorageType)
from checkpoint_schedules import Revolve


class CheckpointError(RuntimeError):
    pass


class Mode(Enum):
    """The mode of the checkpoint manager."""
    RECORD = 1
    FINISHED_RECORDING = 2
    EVALUATED = 3
    EXHAUSTED = 4
    RECOMPUTE = 5
    EVALUATE_ADJOINT = 6


class AdjointSchedule(list):
    """A list of checkpoint actions with a step count.

    Attributes
    ----------
    steps : int
        The number of forward steps in the initial forward calculation.
    """
    def __init__(self, cp_action=(), steps=0):
        # Steps is only intended to provide a rough indicator for progress
        # bars.
        super().__init__(cp_action)
        self.steps = steps


def process_schedule(schedule):
    """Process a checkpoint schedule into forward and adjoint schedules.

    Parameters
    ----------
    schedule : checkpoint_schedules
        A schedule provided by the checkpoint_schedules package.

    Returns
    -------
    AdjointSchedule, AdjointSchedule
        The forward and adjoint schedules.
    """
    schedule = list(schedule)
    index = 0
    forward_steps = 0

    while not isinstance(schedule[index], EndForward):
        if isinstance(schedule[index], Forward):
            forward_steps += len(schedule[index])
        index += 1
    end_forward = index + 1
    reverse_steps = 0

    while not isinstance(schedule[index], EndReverse):
        if isinstance(schedule[index], Reverse) or isinstance(schedule[index], Forward):  # noqa: E501
            reverse_steps += len(schedule[index])
        index += 1
    forward = AdjointSchedule(schedule[:end_forward], forward_steps)
    reverse = AdjointSchedule(schedule[end_forward:], reverse_steps)

    return forward, reverse


class CheckpointManager:
    """Manage the executions of the forward and adjoint solvers.

    Attributes
    ----------
    schedule : checkpoint_schedules.schedule
        A schedule provided by the checkpoint_schedules package.
    tape : Tape
        A list of blocks :class:`Block` instances.
        Each block represents one operation in the forward model.
    max_n : int
        The number of forward steps in the initial forward calculation.
    """
    def __init__(self, schedule, tape):
        # For now, only support Revolve schedules.
        assert isinstance(schedule, Revolve)

        self.fwd_schedule, self.rev_schedule = process_schedule(schedule)
        if schedule.uses_storage_type(StorageType.DISK) and not tape._package_data:  # noqa: E501
            raise CheckpointError(
                "The schedule employs disk checkpointing but it is not configured."  # noqa: E501
            )
        self.tape = tape
        self._schedule = schedule
        self.timesteps = schedule.max_n
        self.mode = Mode.RECORD
        self._iterator = iter(self.fwd_schedule)
        self._current = next(self._iterator)
        self._configuration = None
        self._configuration_step = None
        self._stored_timesteps = []
        self._shedule = schedule
        self._checkpointable_state = set()
        # Tell the tape to only checkpoint input data until told otherwise.
        self.tape.latest_checkpoint = 0
        # Process any initial instructions on the tape.
        self.end_timestep(-1)

    def end_timestep(self, timestep):
        """Process the end of a timestep while recording.

        Parameters
        ----------
        timestep : int
            The number of forward steps in the initial forward calculation.
        """
        if self.mode == Mode.EVALUATED:
            raise CheckpointError("Not enough timesteps in schedule.")
        elif self.mode != Mode.RECORD:
            raise CheckpointError(f"Cannot end timestep in {self.mode}")
        while not self.process_taping(self._current, timestep + 1):
            self._current = next(self._iterator)

    def end_taping(self):
        """Process the end of the forward execution."""
        current_timestep = self.tape.latest_timestep
        while self.mode != Mode.EVALUATED:
            self.end_timestep(current_timestep)
            current_timestep += 1

    @singledispatchmethod
    def process_taping(self, cp_action, timestep):
        """Perform any checkpoint action demanded by schedule at timestep.

        If execution should now continue, return True. Otherwise return
        False and the next cp_action in the schedule will be processed.

        Parameters
        ----------
        cp_action : CheckpointAction
            A schedule provided by the checkpoint_schedules package.
        timestep : int
            The number of forward steps in the initial forward calculation.
        """
        raise CheckpointError(f"Unable to process {cp_action} while taping.")

    @process_taping.register(Forward)
    def _(self, cp_action, timestep):
        if timestep < (cp_action.n0):
            raise CheckpointError(
                "Timestep is before start of Forward action."
            )

        self._configuration = cp_action
        self._configuration_step = timestep
        self.tape._eagerly_checkpoint_outputs = False
        n1 = min(cp_action.n1, self.timesteps)

        if timestep in cp_action and timestep < n1:
            self.tape.get_blocks().append_step()

            if timestep == cp_action.n0 and cp_action.write_ics:
                for data in self.tape._package_data.values():
                    data.configure_checkpointing(cp_action.storage)
                self.tape.latest_checkpoint = timestep
                self._configuration = cp_action
                self._configuration_step = timestep
                self.tape.timesteps[cp_action.n0].checkpoint()
            return True
        else:
            return False

    @process_taping.register(EndForward)
    def _(self, cp_action, timestep):
        print(len(self.tape.timesteps))
        # The correct number of forward steps has been taken
        self.mode = Mode.EVALUATED
        return True

    def recompute(self, functional=None):
        """Recompute the forward model."""
        self.mode = Mode.RECOMPUTE

        with self.tape.progress_bar("Evaluating Functional",
                                    max=self.fwd_schedule.steps) as bar:
            for cp_action in self.fwd_schedule:
                self.process_operation(cp_action, bar, functional=functional)

    def evaluate_adj(self, last_block, markings):
        """Evaluate the adjoint model.

        Parameters
        ----------
        last_block : int
            The last block to be evaluated.
        markings : dict
            A dictionary of variable markings.
        """
        # Work out other cases when they arise.
        assert last_block == 0
        if self.mode == Mode.RECORD:
            # The declared timesteps were not exhausted while taping.
            self.end_taping()

        if self.mode not in (Mode.EVALUATED, Mode.FINISHED_RECORDING):
            raise CheckpointError("Evaluate Functional before calling gradient.")  # noqa: E501

        with self.tape.progress_bar("Evaluating Adjoint",
                                    max=self.rev_schedule.steps) as bar:
            for cp_action in list(self.rev_schedule):
                self.process_operation(cp_action, bar, markings=markings)
                # Only set the mode after the first backward in order to handle
                # that step correctly.
                self.mode = Mode.EVALUATE_ADJOINT

    @singledispatchmethod
    def process_operation(self, operation, bar, **kwargs):
        """Perform the operations required by the schedule."""
        raise CheckpointError(f"Unable to process {operation}.")

    @process_operation.register(Forward)
    def _(self, cp_action, bar, functional=None, **kwargs):
        n1 = min(cp_action.n1, self.timesteps)
        for step in range(n1):
            bar.next()
            current_step = self.tape.timesteps[step]
            self._stored_timesteps.append(step)
            for block in current_step:
                block.recompute()
            if not self._configuration.write_adj_deps:
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
    def _(self, cp_action, bar, markings, functional=None, **kwargs):
        for step in cp_action:
            bar.next()
            current_step = self.tape.timesteps[step]
            for block in reversed(current_step):
                block.evaluate_adj(markings=markings)
            # Output variables are used for the last time when running
            # backwards.
            for block in current_step:
                for var in block.get_outputs():
                    var.checkpoint = None
                    var.reset_variables(("tlm",))
                    if not var.is_control:
                        var.reset_variables(("adjoint", "hessian"))
        if cp_action.clear_adj_deps:
            to_keep = self._checkpointable_state
            if functional:
                to_keep = to_keep.union([functional.block_variable])
            for step in self._stored_timesteps:
                for block in self.tape.timesteps[step]:
                    for output in block.get_outputs():
                        if output not in to_keep:
                            output.checkpoint = None
        self._checkpointable_state = set()

    @process_operation.register(Copy)
    def _(self, cp_action, bar, **kwargs):
        self.rev_schedule.append(cp_action)
        current_step = self.tape.timesteps[cp_action.n]
        current_step.restore_from_checkpoint()
        self._checkpointable_state = current_step.checkpointable_state

    @process_operation.register(Move)
    def _(self, cp_action, bar, **kwargs):
        self.rev_schedule.append(cp_action)
        current_step = self.tape.timesteps[cp_action.n]
        current_step.restore_from_checkpoint()
        self._checkpointable_state = current_step.checkpointable_state
        current_step.delete_checkpoint()

    @process_operation.register(EndForward)
    def _(self, cp_action, bar, **kwargs):
        self.mode = Mode.EVALUATED

    @process_operation.register(EndReverse)
    def _(self, cp_action, bar, **kwargs):
        print(len(self.tape.timesteps))
        if self._schedule.is_exhausted:
            self.mode = Mode.EXHAUSTED
        else:
            self.mode = Mode.EVALUATED