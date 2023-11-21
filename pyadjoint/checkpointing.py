from enum import Enum
from functools import singledispatchmethod
from checkpoint_schedules import (
    Copy, Move, EndForward, EndReverse, Forward, Reverse, StorageType)
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
    cp_action : list
        A list of checkpoint actions from a schedule provided by the
        checkpoint_schedules package.
    steps : int
        The number of forward steps in the initial forward calculation.

    Notes
    -----
    `steps` is only intended to provide a rough indicator for progress bars.
    """
    def __init__(self, cp_action=(), steps=0):
        super().__init__(cp_action)
        self.steps = steps


def process_schedule(schedule):
    """Process a checkpoint schedule into forward and adjoint schedules.

    Parameters
    ----------
    schedule : checkpoint_schedules.schedule
        A schedule provided by the checkpoint_schedules package.

    Examples
    --------
    >>> from checkpoint_schedules import Revolve
    >>> max_n = 3
    >>> snaps_in_ram = 1
    >>> schedule = Revolve(max_n, snaps_in_ram)
    The `schedule` is a generator of checkpoint actions.
    When we make a list of the schedule, we get a list of checkpoint actions
    as follows:
    >>> list(schedule)
    [Forward(0, 2, True, False, StorageType.RAM),
    Forward(2, 3, False, True, StorageType.WORK),
    EndForward(),
    Reverse(3, 2, True),
    Copy(0, StorageType.RAM, StorageType.WORK),
    Forward(0, 1, False, False, StorageType.WORK),
    Forward(1, 2, False, True, StorageType.WORK),
    Reverse(2, 1, True)
    ]
    This current function processes the schedule into forward and adjoint list,
    as exemplified below:
    >>> forward, reverse = process_schedule(schedule)
    >>> list(forward)
    [Forward(0, 2, True, False, StorageType.RAM),
    Forward(2, 3, False, True, StorageType.WORK),
    EndForward()]
    >>> list(reverse)
    [Reverse(3, 2, True),
    Copy(0, StorageType.RAM, StorageType.WORK),
    Forward(0, 1, False, False, StorageType.WORK),
    Forward(1, 2, False, True, StorageType.WORK),
    Reverse(2, 1, True)]

    Returns
    -------
    list, list
        Forward and adjoint list of schedules.
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
        if (
            isinstance(schedule[index], Reverse)
            or isinstance(schedule[index], Forward)
        ):
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

    Notes
    -----
    Currently, automated gradient using checkpointing only supports `Revolve` schedules.
    """
    def __init__(self, schedule, tape):
        if not isinstance(schedule, Revolve):
            raise CheckpointError("Only Revolve schedules are supported.")

        self.fwd_schedule, self.rev_schedule = process_schedule(schedule)
        if (
            schedule.uses_storage_type(StorageType.DISK)
            and not tape._package_data
        ):
            raise CheckpointError(
                "The schedule employs disk checkpointing but it is not configured."
            )
        self.tape = tape
        self._schedule = schedule
        self.timesteps = schedule.max_n
        self.mode = Mode.RECORD
        self._iterator = iter(self.fwd_schedule)
        self._current = next(self._iterator)
        # Tell the tape to only checkpoint input data until told otherwise.
        self.tape.latest_checkpoint = 0
        self.end_timestep(-1)

    def end_timestep(self, timestep):
        """Mark the end of one timestep when taping the forward model.

        Parameters
        ----------
        timestep : int
            The current timestep.
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
        """A single-dispatch generic function.
        The process taping is used while the forward model is taped.

        Parameters
        ----------
        cp_action : checkpoint_schedules.CheckpointAction
            A checkpoint action from the schedule. The actions can
            be `Forward`, `Reverse`, `EndForward`, `EndReverse`, `Copy`, or
            `Move`.
        timestep : int
            The current timestep.

        Raises
        ------
        CheckpointError
            If the checkpoint action is not supported.

        Notes
        -----
        Additional details about checkpoint_schedules can be found at
        checkpoint_schedules
        `documentation <https://www.firedrakeproject.org/checkpoint_schedules/>`_.

        Returns
        -------
        bool
            `'True'`, while a forward action is not finalised. Otherwise,
            `'False'`.
        """
        raise CheckpointError(f"Unable to process {cp_action} while taping.")

    @process_taping.register(Forward)
    def _(self, cp_action, timestep):
        if timestep < (cp_action.n0):
            raise CheckpointError(
                "Timestep is before start of Forward action."
            )

        self.tape._eagerly_checkpoint_outputs = cp_action.write_adj_deps

        if timestep > cp_action.n0:
            if cp_action.write_ics and timestep == (cp_action.n0 + 1):
                self.tape.timesteps[cp_action.n0].checkpoint()

            if not cp_action.write_adj_deps:
                for var in self.tape.timesteps[timestep - 1].checkpointable_state:
                    var._checkpoint = None
                for block in self.tape.timesteps[timestep - 1]:
                    for output in block.get_outputs():
                        output._checkpoint = None

        if timestep in cp_action:
            self.tape.get_blocks().append_step()
            if cp_action.write_ics:
                self.tape.latest_checkpoint = cp_action.n0
            return True
        else:
            return False

    @process_taping.register(EndForward)
    def _(self, cp_action, timestep):
        if timestep != self.timesteps:
            raise CheckpointError(
                "The correct number of forward steps has notbeen taken."
            )
        assert timestep == self.timesteps
        self.mode = Mode.EVALUATED
        return True

    def recompute(self, functional=None):
        """Recompute the forward model.

        Parameters
        ----------
        functional : BlockVariable
            The functional to be evaluated.
        """
        self.mode = Mode.RECOMPUTE

        with self.tape.progress_bar("Evaluating Functional",
                                    max=self.fwd_schedule.steps) as bar:
            # Restore the initial condition to advance the forward model
            # from the step 0.
            current_step = self.tape.timesteps[self.fwd_schedule[0].n0]
            current_step.restore_from_checkpoint()
            for cp_action in self.fwd_schedule:
                self.process_operation(cp_action, bar, functional=functional)

    def evaluate_adj(self, last_block, markings):
        """Evaluate the adjoint model.

        Parameters
        ----------
        last_block : int
            The last block to be evaluated.
        markings : bool
            If True, then each `BlockVariable` of the current block will have
            set `marked_in_path` attribute indicating whether their adjoint
            components are relevant for computing the final target adjoint
            values. Default is False.
        """
        # Work out other cases when they arise.
        if last_block != 0:
            raise NotImplementedError(
                "Only the first block can be evaluated at present."
            )

        if self.mode == Mode.RECORD:
            # The declared timesteps were not exhausted while taping.
            self.end_taping()

        if self.mode not in (Mode.EVALUATED, Mode.FINISHED_RECORDING):
            raise CheckpointError("Evaluate Functional before calling gradient.")

        with self.tape.progress_bar("Evaluating Adjoint",
                                    max=self.rev_schedule.steps) as bar:
            for cp_action in list(self.rev_schedule):
                self.process_operation(cp_action, bar, markings=markings)
                # Only set the mode after the first backward in order to handle
                # that step correctly.
                self.mode = Mode.EVALUATE_ADJOINT

    @singledispatchmethod
    def process_operation(self, cp_action, bar, **kwargs):
        """Perform the the checkpoint action required by the schedule.

        Parameters
        ----------
        cp_action : checkpoint_schedules.CheckpointAction
            A checkpoint action from the schedule. The actions can
            be `Forward`, `Reverse`, `EndForward`, `EndReverse`, `Copy`, or
            `Move`.
        bar : progressbar.ProgressBar
            A progress bar to display the progress of the reverse executions.

        Raises
        ------
        CheckpointError
            If the checkpoint action is not supported.
        """
        raise CheckpointError(f"Unable to process {cp_action}.")

    @process_operation.register(Forward)
    def _(self, cp_action, bar, functional=None, **kwargs):
        for step in cp_action:
            bar.next()
            # Get the blocks of the current step.
            current_step = self.tape.timesteps[step]
            for block in current_step:
                block.recompute()

            if cp_action.write_ics:
                if step == cp_action.n0:
                    for var in current_step.checkpointable_state:
                        if var.checkpoint:
                            current_step._checkpoint.update(
                                {var: var.checkpoint}
                            )
                if not cp_action.write_adj_deps:
                    next_step = self.tape.timesteps[step + 1]
                    # The checkpointable state set of the current step.
                    to_keep = next_step.checkpointable_state
                    if functional:
                        # `to_keep` holds informations of the blocks required
                        # for restarting the forward model from a step `n`.
                        to_keep = to_keep.union([functional.block_variable])
                    for block in current_step:
                        for bv in block.get_outputs():
                            if bv not in to_keep:
                                bv._checkpoint = None
                    for var in (current_step.checkpointable_state - to_keep):
                        var._checkpoint = None

    @process_operation.register(Reverse)
    def _(self, cp_action, bar, markings, functional=None, **kwargs):
        for step in cp_action:
            bar.next()
            # Get the blocks of the current step.
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
                    to_keep = current_step.checkpointable_state
                    if functional:
                        to_keep = to_keep.union([functional.block_variable])
                    for output in block.get_outputs():
                        if output not in to_keep:
                            output._checkpoint = None

    @process_operation.register(Copy)
    def _(self, cp_action, bar, **kwargs):
        current_step = self.tape.timesteps[cp_action.n]
        current_step.restore_from_checkpoint()

    @process_operation.register(Move)
    def _(self, cp_action, bar, **kwargs):
        current_step = self.tape.timesteps[cp_action.n]
        current_step.restore_from_checkpoint()
        current_step.delete_checkpoint()

    @process_operation.register(EndForward)
    def _(self, cp_action, bar, **kwargs):
        self.mode = Mode.EVALUATED

    @process_operation.register(EndReverse)
    def _(self, cp_action, bar, **kwargs):
        if self._schedule.is_exhausted:
            self.mode = Mode.EXHAUSTED
        else:
            self.mode = Mode.EVALUATED
