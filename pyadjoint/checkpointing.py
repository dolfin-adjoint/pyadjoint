from enum import Enum
import sys
import gc
import logging
from functools import singledispatchmethod
from checkpoint_schedules import Copy, Move, EndForward, EndReverse, \
    Forward, Reverse, StorageType, SingleMemoryStorageSchedule
# A callback interface allowing the user to provide a
# custom error message when disk checkpointing is not configured.
disk_checkpointing_callback = {}


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
        gc_timestep_frequency (None or int): The number of timesteps between garbage collections. The default
        is `None`, which means no invoking the garbage collector during the executions. If an integer is
        provided, the garbage collector is applied every `gc_timestep_frequency` timestep. That is useful when
        being affected by the Python fails to track and clean all checkpoint objects in memory properly.
        gc_generation (int): The generation for garbage collection. Default is 2 that runs a full collection.
        To have more information about the garbage collector generation,
        please refer to the `documentation
        <https://docs.python.org/3/library/gc.html#gc.collect>`_.

    Attributes:
        tape (Tape): A list of blocks :class:`Block` instances.
        _schedule (checkpoint_schedules.schedule): A schedule provided by the `checkpoint_schedules` package.
        forward_schedule (list): A list of `checkpoint_schedules` actions used to manage the execution of the
        forward model.
        reverse_schedule (list): A list of `checkpoint_schedules` actions used to manage the execution of the
        reverse model.
        total_timesteps (int): The total number of timesteps to execute the forward model.
        mode (Mode): The mode of the checkpoint manager. The possible modes are `RECORD`, `FINISHED_RECORDING`,
        `EVALUATED`, `EXHAUSTED`, `RECOMPUTE`, and `EVALUATE_ADJOINT`. Additional information about the modes
        can be found class:`Mode`.
        _current_action (checkpoint_schedules.CheckpointAction): The current `checkpoint_schedules` action.

    """
    def __init__(self, schedule, tape, gc_timestep_frequency=None, gc_generation=2):
        if (
            schedule.uses_storage_type(StorageType.DISK)
            and not tape._package_data
        ):
            raise CheckpointError(
                "The schedule employs disk checkpointing but it is not configured.\n"
                + "\n".join(disk_checkpointing_callback.values()))
        self.tape = tape
        self._schedule = schedule
        self.forward_schedule = []
        self.reverse_schedule = []
        if self._schedule.max_n:
            self.total_timesteps = schedule.max_n
        else:
            # We have schedules in `checkpoint_schedules` offering the flexibility to determine
            # the desired steps during the forward execution. For this type of schedule, we do not
            # have the total number of timesteps `self._schedule.max_n`. Therefore, we set the
            # `self.total_timesteps` to the maximum value of the `sys.maxsize` to indicate that
            # the total number of timesteps is not known.
            self.total_timesteps = sys.maxsize
        self.mode = Mode.RECORD
        self._current_action = next(self._schedule)
        self.forward_schedule.append(self._current_action)
        # Tell the tape to only checkpoint input data until told otherwise.
        self.tape.latest_checkpoint = 0
        self._keep_init_state_in_work = False
        self._gc_timestep_frequency = gc_timestep_frequency
        self._gc_generation = gc_generation
        # ``self._global_deps`` stores checkpoint dependencies that remain unchanged across
        # timesteps (``self.tape.timesteps``). During the forward taping process, the code
        # checks whether a dependency is in ``self._global_deps`` to avoid unnecessary clearing
        # and recreation of its checkpoint data.
        self._global_deps = set()
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
        if not self._schedule.max_n:
            # Inform the schedule that the forward model has finished.
            self._schedule.finalize(len(self.tape.timesteps))
            # `self._schedule.finalize` updates `self._schedule.max_n`.
            self.total_timesteps = self._schedule.max_n
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
        _store_checkpointable_state = False
        _store_adj_dependencies = False
        if timestep > cp_action.n0 and cp_action.storage != StorageType.WORK:
            if cp_action.write_ics and timestep == (cp_action.n0 + 1):
                # Store the checkpoint data. This is the required data for
                # restarting the forward model from the step `n0`.
                _store_checkpointable_state = True
            if cp_action.write_adj_deps:
                # Store the checkpoint data. This is the required data for
                # computing the adjoint model from the step `n1`.
                _store_adj_dependencies = True
            if (
                (_store_checkpointable_state or _store_adj_dependencies)
                and cp_action.storage == StorageType.DISK
            ):
                for package in self.tape._package_data.values():
                    package.continue_checkpointing()
            if timestep == 1:
                # Store the possible global dependencies.
                for deps in self.tape.timesteps[timestep - 1].checkpointable_state:
                    self._global_deps.add(deps)
            else:
                # Check if the block variables stored in `self._global_deps` are still
                # dependencies in the previous timestep. If not, remove them from the
                # global dependencies.
                deps_to_clear = self._global_deps.difference(
                    self.tape.timesteps[timestep - 1].checkpointable_state)

                # Remove the block variables that are not global dependencies.
                self._global_deps.difference_update(deps_to_clear)

                # For no global dependencies, checkpoint storage occurs at a self.tape
                # timestep only when required by an action from the schedule. Thus, we
                # have to clear the checkpoint of block variables excluded from the self._global_deps.
                for deps in deps_to_clear:
                    deps._checkpoint = None

            self.tape.timesteps[timestep - 1].checkpoint(
                _store_checkpointable_state, _store_adj_dependencies, self._global_deps)
            # Remove unnecessary variables in working memory from previous steps.
            for var in self.tape.timesteps[timestep - 1].checkpointable_state - self._global_deps:
                var._checkpoint = None

            for block in self.tape.timesteps[timestep - 1]:
                for out in block.get_outputs():
                    out._checkpoint = None

        if cp_action.storage == StorageType.DISK:
            # Activate disk checkpointing only in the checkpointing process.
            for package in self.tape._package_data.values():
                package.pause_checkpointing()

        if isinstance(self._gc_timestep_frequency, int) and timestep % self._gc_timestep_frequency == 0:
            logging.info("Running a garbage collection cycle")
            gc.collect(self._gc_generation)

        if timestep in cp_action and timestep < self.total_timesteps:
            self.tape.get_blocks().append_step()
            if cp_action.write_ics:
                self.tape.latest_checkpoint = cp_action.n0
            return True
        else:
            return False

    @process_taping.register(EndForward)
    def _(self, cp_action, timestep):
        if timestep != self.total_timesteps:
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
        if self.mode == Mode.RECORD:
            # Finalise the taping process.
            self.end_taping()
        if self._schedule.uses_storage_type(StorageType.DISK):
            # Clear the data of the current state before recomputing.
            for package in self.tape._package_data.values():
                package.reset()
        self.mode = Mode.RECOMPUTE
        with self.tape.progress_bar("Evaluating Functional", max=self.total_timesteps) as progress_bar:
            # Restore the initial condition to advance the forward model from the step 0.
            current_step = self.tape.timesteps[self.forward_schedule[0].n0]
            current_step.restore_from_checkpoint(self.forward_schedule[0].storage)
            for cp_action in self.forward_schedule:
                self._current_action = cp_action
                self.process_operation(cp_action, progress_bar, functional=functional)

    def evaluate_adj(self, last_block, markings):
        """Evaluate the adjoint model.

        Args:
            last_block (int): The last block to be evaluated.
            markings (bool): If `True`, then each `BlockVariable` of the current block will have set
            `is_control_dependent` attribute indicating whether their adjoint components are relevant
            for computing the final target adjoint values and `is_functional_dependency` indicating
            whether the functional depends on their value.
        """
        # Work out other cases when they arise.
        if last_block != 0:
            raise NotImplementedError(
                "Only the first block can be evaluated at present."
            )
        if self.mode == Mode.RECORD:
            # Finalise the taping process.
            self.end_taping()

        if self.mode not in (Mode.EVALUATED, Mode.FINISHED_RECORDING):
            raise CheckpointError("Evaluate Functional before calling gradient.")

        with self.tape.progress_bar("Evaluating Adjoint", max=self.total_timesteps) as progress_bar:
            if self.reverse_schedule:
                for cp_action in self.reverse_schedule:
                    self.process_operation(cp_action, progress_bar, markings=markings)
            else:
                while not isinstance(self._current_action, EndReverse):
                    cp_action = next(self._schedule)
                    self._current_action = cp_action
                    self.reverse_schedule.append(cp_action)
                    self.process_operation(cp_action, progress_bar, markings=markings)

            # Only set the mode after the first backward in order to handle
            # that step correctly.
            self.mode = Mode.EVALUATE_ADJOINT

    @singledispatchmethod
    def process_operation(self, cp_action, progress_bar, **kwargs):
        """A function used to process the forward and adjoint executions.
        This single-dispatch generic function is used in the `Blocks`
        recomputation and adjoint evaluation with checkpointing.

        Note:
            The documentation of the `checkpoint_schedules` actions is available
            `here <https://www.firedrakeproject.org/checkpoint_schedules/>`_.

        Args:
            cp_action (checkpoint_schedules.CheckpointAction): A checkpoint action obtained from the
            `checkpoint_schedules`.
            progress_bar (progressbar.ProgressBar): A progress bar to display the progress of the reverse executions.
            kwargs: Additional keyword arguments.

        Raises:
            CheckpointError: If the checkpoint action is not supported.
        """
        raise CheckpointError(f"Unable to process {cp_action}.")

    @process_operation.register(Forward)
    def _(self, cp_action, progress_bar, functional=None, **kwargs):
        step = cp_action.n0
        #  In a dynamic schedule `cp_action` can be unbounded so we also need
        # to check `self.total_timesteps`.
        while step in cp_action and step < self.total_timesteps:
            if self.mode == Mode.RECOMPUTE and progress_bar:
                progress_bar.next()
            # Get the blocks of the current step.
            current_step = self.tape.timesteps[step]
            for block in current_step:
                block.recompute()
            _store_checkpointable_state = False
            _store_adj_dependencies = False
            if cp_action.storage != StorageType.WORK:
                if (cp_action.write_ics and step == cp_action.n0):
                    _store_checkpointable_state = True
                if cp_action.write_adj_deps:
                    _store_adj_dependencies = True
                if (
                    (_store_checkpointable_state or _store_adj_dependencies)
                    and cp_action.storage == StorageType.DISK
                ):
                    for package in self.tape._package_data.values():
                        package.continue_checkpointing()
                current_step.checkpoint(
                    _store_checkpointable_state, _store_adj_dependencies, self._global_deps)

            to_keep = set()
            if step < (self.total_timesteps - 1):
                next_step = self.tape.timesteps[step + 1]
                # The checkpointable state set of the current step.
                to_keep = next_step.checkpointable_state
            if functional:
                to_keep = to_keep.union([functional.block_variable])

            for var in current_step.checkpointable_state - to_keep.union(self._global_deps):
                # Handle the case where step is 0
                if step == 0 and var not in current_step._checkpoint:
                    # Ensure initialisation state is kept.
                    self._keep_init_state_in_work = True
                    break

                # Handle the case for SingleMemoryStorageSchedule
                if isinstance(self._schedule, SingleMemoryStorageSchedule):
                    if step > 1 and var not in self.tape.timesteps[step - 1].adjoint_dependencies:
                        var._checkpoint = None
                    continue

                # Handle variables in the initial timestep
                if (
                    var in self.tape.timesteps[0].checkpointable_state
                    and self._keep_init_state_in_work
                ):
                    continue

                # Clear the checkpoint for other cases
                var._checkpoint = None

            for block in current_step:
                # Remove unnecessary variables from previous steps.
                for bv in block.get_outputs():
                    if (
                        (cp_action.write_adj_deps and cp_action.storage != StorageType.WORK)
                        or not cp_action.write_adj_deps
                    ):
                        if bv not in to_keep:
                            bv._checkpoint = None
                    else:
                        if bv not in current_step.adjoint_dependencies.union(to_keep):
                            bv._checkpoint = None

            if self._gc_timestep_frequency and step % self._gc_timestep_frequency == 0:
                logging.info("Running a garbage collection cycle")
                gc.collect(self._gc_generation)

            step += 1
            if cp_action.storage == StorageType.DISK:
                # Activate disk checkpointing only in the checkpointing process.
                for package in self.tape._package_data.values():
                    package.pause_checkpointing()

    @process_operation.register(Reverse)
    def _(self, cp_action, progress_bar, markings, functional=None, **kwargs):
        for step in cp_action:
            if progress_bar:
                progress_bar.next()
            # Get the blocks of the current step.
            current_step = self.tape.timesteps[step]
            for block in reversed(current_step):
                block.evaluate_adj(markings=markings)
                if not current_step._revised_adj_deps:
                    # Update the adjoint dependency set.
                    for deps in block.get_dependencies():
                        if deps.is_functional_dependency:
                            current_step.adjoint_dependencies.add(deps)
                    for out in block._outputs:
                        if not out.is_functional_dependency:
                            current_step.adjoint_dependencies.discard(out)
            current_step._revised_adj_deps = True
            # Output variables are used for the last time when running
            # backwards.
            to_keep = current_step.checkpointable_state
            if functional:
                to_keep = to_keep.union([functional.block_variable])
            for block in current_step:
                block.reset_adjoint_state()
                for out in block.get_outputs():
                    out.reset_variables(("tlm",))
                    if not out.is_control:
                        out.reset_variables(("adjoint", "hessian"))
                    if cp_action.clear_adj_deps and out not in to_keep:
                        out._checkpoint = None
            if self._gc_timestep_frequency and step % self._gc_timestep_frequency == 0:
                logging.info("Running a garbage collection cycle")
                gc.collect(self._gc_generation)

    @process_operation.register(Copy)
    def _(self, cp_action, progress_bar, **kwargs):
        current_step = self.tape.timesteps[cp_action.n]
        current_step.restore_from_checkpoint(cp_action.from_storage)

    @process_operation.register(Move)
    def _(self, cp_action, progress_bar, **kwargs):
        current_step = self.tape.timesteps[cp_action.n]
        current_step.restore_from_checkpoint(cp_action.from_storage)
        current_step.delete_checkpoint()

    @process_operation.register(EndForward)
    def _(self, cp_action, progress_bar, **kwargs):
        self.mode = Mode.EVALUATED

    @process_operation.register(EndReverse)
    def _(self, cp_action, progress_bar, **kwargs):
        if self._schedule.is_exhausted:
            self.mode = Mode.EXHAUSTED
        else:
            self.mode = Mode.EVALUATED
