from enum import Enum
from functools import singledispatchmethod
from checkpoint_schedules import (
    Copy, Move, EndForward, EndReverse, Forward, Reverse, StorageType)
from checkpoint_schedules import Revolve, MultistageCheckpointSchedule


class CheckpointError(RuntimeError):
    pass


class Mode(Enum):
    """The mode of the checkpoint manager.

    RECORD: The forward model is being taped.
    FINISHED_RECORDING: The forward model is finished being taped.
    EVALUATED: The forward model was evaluated.
    EXHAUSTED: The forward and the adjoint models were evaluated and the schedule has concluded.
    RECOMPUTE: The forward model is being recomputed.
    EVALUATE_ADJOINT: The adjoint model is being evaluated.

    """
    RECORD = 1
    FINISHED_RECORDING = 2
    EVALUATED = 3
    EXHAUSTED = 4
    RECOMPUTE = 5
    EVALUATE_ADJOINT = 6


class CheckpointManager:
    """Manage the executions of the forward and adjoint solvers.

    Args:
        schedule (checkpoint_schedules.schedule): A schedule provided by the `checkpoint_schedules` package.
        tape (Tape): A list of blocks :class:`Block` instances.

    Attributes:
        tape (Tape): A list of blocks :class:`Block` instances.
        _schedule (checkpoint_schedules.schedule): A schedule provided by the `checkpoint_schedules` package.
        forward_schedule (list): A list of `checkpoint_schedules` actions used to manage the execution of the
        forward model.
        reverse_schedule (list): A list of `checkpoint_schedules` actions used to manage the execution of the
        reverse model.
        timesteps (int): The initial number of timesteps.
        adjoint_evaluated (bool): A boolean indicating whether the adjoint model has been evaluated.
        mode (Mode): The mode of the checkpoint manager. The possible modes are `RECORD`, `FINISHED_RECORDING`,
        `EVALUATED`, `EXHAUSTED`, `RECOMPUTE`, and `EVALUATE_ADJOINT`. Additional information about the modes
        can be found class:`Mode`.
        _current_action (checkpoint_schedules.CheckpointAction): The current `checkpoint_schedules` action.

    """
    def __init__(self, schedule, tape):
        if (
            not isinstance(schedule, Revolve)
            and not isinstance(schedule, MultistageCheckpointSchedule)
        ):
            raise CheckpointError(
                "Only Revolve and MultistageCheckpointSchedule schedules are supported."
            )

        if (
            schedule.uses_storage_type(StorageType.DISK)
            and not tape._package_data
        ):
            raise CheckpointError(
                "The schedule employs disk checkpointing but it is not configured."
            )
        self.tape = tape
        self._schedule = schedule
        self.forward_schedule = []
        self.reverse_schedule = []
        self.timesteps = schedule.max_n
        # This variable is used to indicate whether the adjoint model has been evaluated at the checkpoint.
        self.adjoint_evaluated = False
        self.mode = Mode.RECORD
        self._current_action = next(self._schedule)
        self.forward_schedule.append(self._current_action)
        # Tell the tape to only checkpoint input data until told otherwise.
        self.tape.latest_checkpoint = 0
        self.end_timestep(-1)

    def end_timestep(self, timestep):
        """Mark the end of one timestep when taping the forward model.

        Args:
            timestep (int): The current timestep.
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
        while self.mode != Mode.EVALUATED:
            self.end_timestep(current_timestep)
            current_timestep += 1

    @singledispatchmethod
    def process_taping(self, cp_action, timestep):
        """Implement checkpointing schedule actions while taping.
        
        A single-dispatch generic function.

        Note:
            To have more information about the `checkpoint_schedules`, please refer to the
            `documentation <https://www.firedrakeproject.org/checkpoint_schedules/>`_.
            Detailed descriptions of the actions used in the process taping can be found at the following links:
            `Forward <https://www.firedrakeproject.org/checkpoint_schedules/checkpoint_schedules.html#
            checkpoint_schedules.schedule.Forward>`_ and `End_Forward <https://www.firedrakeproject.org/
            checkpoint_schedules/checkpoint_schedules.html#checkpoint_schedules.schedule.EndForward>`_.

        Args:
            cp_action (checkpoint_schedules.CheckpointAction): A checkpoint action obtained from the
            `checkpoint_schedules`.
            timestep (int): The current timestep.

        Returns:
            bool: Returns `True` if the timestep is in the `checkpoint_schedules` action.
            For example, if the `checkpoint_schedules` action is `Forward(0, 4, True, False, StorageType.DISK)`,
            then timestep `0, 1, 2, 3` is considered within the action; timestep `4` is not considered within the
            action and `False` is returned.

        Raises:
            CheckpointError: If the checkpoint action is not supported.
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
                # Stores the checkpoint data in RAM
                # This data will be used to restart the forward model
                # from the step `n0` in the reverse computations.
                self.tape.timesteps[cp_action.n0].checkpoint()

            if not cp_action.write_adj_deps:
                # Remove unnecessary variables from previous steps.
                for var in self.tape.timesteps[timestep - 1].checkpointable_state:
                    var._checkpoint = None
                for block in self.tape.timesteps[timestep - 1]:
                    # Remove unnecessary variables from previous steps.
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
        self.mode = Mode.EVALUATED
        return True

    def recompute(self, functional=None):
        """Recompute the forward model.

        Args:
            functional (BlockVariable): The functional to be evaluated.
        """
        self.mode = Mode.RECOMPUTE

        with self.tape.progress_bar("Evaluating Functional",
                                    max=self.timesteps) as bar:
            # Restore the initial condition to advance the forward model
            # from the step 0.
            current_step = self.tape.timesteps[self.forward_schedule[0].n0]
            current_step.restore_from_checkpoint()
            for cp_action in self.forward_schedule:
                self._current_action = cp_action
                self.process_operation(cp_action, bar, functional=functional)

    def evaluate_adj(self, last_block, markings):
        """Evaluate the adjoint model.

        Args:
            last_block (int): The last block to be evaluated.
            markings (bool): If `True`, then each `BlockVariable` of the current block will have set
            `marked_in_path` attribute indicating whether their adjoint components are relevant for
            computing the final target adjoint values.
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
                                    max=self.timesteps) as bar:
            if self.adjoint_evaluated:
                reverse_iterator = iter(self.reverse_schedule)
            while not isinstance(self._current_action, EndReverse):
                if not self.adjoint_evaluated:
                    self._current_action = next(self._schedule)
                    self.reverse_schedule.append(self._current_action)
                else:
                    self._current_action = next(reverse_iterator)
                self.process_operation(self._current_action, bar, markings=markings)
                # Only set the mode after the first backward in order to handle
                # that step correctly.
                self.mode = Mode.EVALUATE_ADJOINT

            # Inform that the adjoint model has been evaluated.
            self.adjoint_evaluated = True

    @singledispatchmethod
    def process_operation(self, cp_action, bar, **kwargs):
        """A function used to process the forward and adjoint executions.
        This single-dispatch generic function is used in the `Blocks`
        recomputation and adjoint evaluation with checkpointing.

        Note:
            The documentation of the `checkpoint_schedules` actions is available
            `here <https://www.firedrakeproject.org/checkpoint_schedules/>`_.

        Args:
            cp_action (checkpoint_schedules.CheckpointAction): A checkpoint action obtained from the
            `checkpoint_schedules`.
            bar (progressbar.ProgressBar): A progress bar to display the progress of the reverse executions.
            kwargs: Additional keyword arguments.

        Raises:
            CheckpointError: If the checkpoint action is not supported.
        """
        raise CheckpointError(f"Unable to process {cp_action}.")

    @process_operation.register(Forward)
    def _(self, cp_action, bar, functional=None, **kwargs):
        for step in cp_action:
            if self.mode == Mode.RECOMPUTE:
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
                        # Remove unnecessary variables from previous steps.
                        for bv in block.get_outputs():
                            if bv not in to_keep:
                                bv._checkpoint = None
                    # Remove unnecessary variables from previous steps.
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
