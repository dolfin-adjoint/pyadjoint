from enum import Enum
import sys
from functools import singledispatchmethod
from checkpoint_schedules import *


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
    Currently, automated gradient using checkpointing only supports `Revolve`
    and `MultistageCheckpointSchedule` schedules.
    """
    def __init__(self, schedule, tape):
        # if (
        #     schedule.uses_storage_type(StorageType.DISK)
        #     and not tape._package_data
        # ):
        #     raise CheckpointError(
        #         "The schedule employs disk checkpointing but it is not configured."
        #     )
        self.tape = tape
        self._schedule = schedule
        self.forward_schedule = []
        self.reverse_schedule = []
        # The total number of timesteps in the forward model.
        if not self._schedule.max_n:
            self.total_steps = sys.maxsize
        else:
            self.total_steps = self._schedule.max_n
        self.mode = Mode.RECORD
        self._current_action = next(self._schedule)
        self.forward_schedule.append(self._current_action)
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
        while not self.process_taping(self._current_action, timestep + 1):
            self._current_action = next(self._schedule)
            self.forward_schedule.append(self._current_action)

    def end_taping(self):
        """Process the end of the forward execution."""
        current_timestep = self.tape.latest_timestep
        if not self._schedule.max_n:
            # Inform the total number of timesteps in the forward model
            self._schedule.finalize(len(self.tape.timesteps))
            self.total_steps = self._schedule.max_n
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
            `True`, while a forward action is not finalised. Otherwise,
            `False`.
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
                # Store the checkpointint data in RAM or on disk.
                # The data will be used to restart the forward model from the
                # step `n0` in the reverse computations.
                self.tape.timesteps[timestep - 1].checkpoint()
            if cp_action.write_adj_deps and cp_action.storage != StorageType.WORK:
                # Stores the checkpointing data in RAM or on disk.
                # The foward data will be used in the adjoint computations.
                if isinstance(self._schedule, SingleDiskStorageSchedule):
                    self.tape.timesteps[timestep - 1].checkpoint()
                else:
                    self.tape.timesteps[cp_action.n1 - 1].checkpoint()
            if cp_action.storage != StorageType.WORK:
                # Remove unnecessary variables from previous steps.
                for var in self.tape.timesteps[timestep - 1].checkpointable_state:
                    var._checkpoint = None
                for block in self.tape.timesteps[timestep - 1]:
                    for output in block.get_outputs():
                        output._checkpoint = None

        if timestep in cp_action and timestep < self.total_steps:
            self.tape.get_blocks().append_step()
            # Get the storage type of every single timestep.
            self.tape.timesteps[timestep]._storage_type = StorageType.NONE
            if cp_action.write_ics and timestep == cp_action.n0:
                self.tape.timesteps[timestep]._storage_type = cp_action.storage
            if cp_action.write_adj_deps:
                self.tape.timesteps[timestep]._storage_type = cp_action.storage
            return True
        else:
            return False

    @process_taping.register(EndForward)
    def _(self, cp_action, timestep):
        if timestep != self.total_steps:
            raise CheckpointError(
                "The correct number of forward steps has notbeen taken."
            )
        self.mode = Mode.EVALUATED
        return True

    def recompute(self, functional=None):
        """Recompute the forward model.

        Parameters
        ----------
        functional : BlockVariable
            The functional to be evaluated.
        """
        if self.mode == Mode.RECORD:
            # The declared timesteps were not exhausted while taping.
            self.end_taping()

        self.mode = Mode.RECOMPUTE
        with self.tape.progress_bar("Evaluating Functional",
                                    max=self.total_steps) as bar:
            # Restore the initial condition to advance the forward model
            # from the step 0.
            self.tape._recomputation = True
            current_step = self.tape.timesteps[self.forward_schedule[0].n0]
            current_step.restore_from_checkpoint()
            for cp_action in self.forward_schedule:
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
            values.
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
                                    max=self.total_steps) as bar:
            if self.reverse_schedule:
                for cp_action in self.reverse_schedule:
                    self.process_operation(cp_action, bar, markings=markings)
            else:
                while not isinstance(self._current_action, EndReverse):
                    cp_action = next(self._schedule)
                    self._current_action = cp_action
                    self.reverse_schedule.append(cp_action)
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
        step = cp_action.n0
        while step in cp_action and step < self.total_steps:
            if self.mode == Mode.RECOMPUTE:
                bar.next()
            # Get the blocks of the current step.
            current_step = self.tape.timesteps[step]
            for block in current_step:
                block.recompute()
            if (
                (cp_action.write_ics and step == cp_action.n0)
                or (cp_action.write_adj_deps
                    and cp_action.storage != StorageType.WORK)
            ):
                for var in current_step.checkpointable_state:
                    if var.checkpoint:
                        current_step._checkpoint.update(
                            {var: var.checkpoint}
                        )
            if cp_action.storage != StorageType.WORK:
                if step < (self.total_steps - 1):
                    next_step = self.tape.timesteps[step + 1]
                    # The checkpointable state set of the current step.
                    to_keep = next_step.checkpointable_state
                if functional:
                    # `to_keep` holds informations of the blocks required
                    # for restarting the forward model from a step `n`.
                    if to_keep:
                        to_keep = to_keep.union([functional.block_variable])
                    else:
                        to_keep = {functional.block_variable}
                for block in current_step:
                    # Remove unnecessary variables from previous steps.
                    for bv in block.get_outputs():
                        if bv not in to_keep:
                            bv._checkpoint = None
                # Remove unnecessary variables from previous steps.
                for var in (current_step.checkpointable_state - to_keep):
                    var._checkpoint = None
            step += 1

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
