# Type dependencies
import os
import re
import threading
from contextlib import contextmanager, ContextDecorator
from itertools import chain
from typing import Optional, Iterable
from abc import ABC, abstractmethod
from .checkpointing import CheckpointManager, CheckpointError, StorageType
from .ordered_set import OrderedSet

_working_tape = None
_annotation_enabled = False


def get_working_tape():
    return _working_tape


def pause_annotation():
    """Switch off annotation."""
    global _annotation_enabled
    _annotation_enabled = False


def continue_annotation():
    """Switch on annotation."""
    global _annotation_enabled
    _annotation_enabled = True
    return _annotation_enabled


class set_working_tape(ContextDecorator):
    """Set a new tape as the working tape.

    This class can be used in three ways:
       1) as a free function to replace the working tape,
       2) as a context manager within which a new tape is set as the working tape,
       3) as a function decorator so that the new tape is set only inside the function.

       Example usage:

        1) Set a new tape as the working tape:

            .. highlight:: python
            .. code-block:: python

                set_working_tape(Tape())

        2) Set a local tape within a context manager:

            .. highlight:: python
            .. code-block:: python

                with set_working_tape() as tape:
                    ...

        3) Set the local tape inside a decorated function.
           The two functions below are equivalent:

            .. highlight:: python
            .. code-block:: python

                @set_working_tape()
                def decorated_function(*args, **kwargs):
                    # do something here
                    return ReducedFunctional(functional, control)

                def context_function(*args, **kwargs):
                    with set_working_tape():
                        # do something here
                        return ReducedFunctional(functional, control)

    """

    def __init__(self, tape=None, **tape_kwargs):
        # Get working tape
        global _working_tape
        # Store current tape
        self.old_tape = _working_tape
        # Set new tape
        self.tape = tape or Tape(**tape_kwargs)
        _working_tape = self.tape

    def __enter__(self):
        return self.tape

    def __exit__(self, *args):
        # Re-establish the original tape
        global _working_tape
        _working_tape = self.old_tape


class stop_annotating(ContextDecorator):
    """A context manager and function decorator within which annotation is stopped.

    Args:
        modifies (OverloadedType or list[OverloadedType]): One or more
            variables which appear in the tape and whose values are to be
            changed inside the context manager.

    The `modifies` argument is intended to be used by user code which
    changes the value of inputs to the adjoint calculation such as time varying
    forcings. Its effect is to create a new block variable for each of the
    modified variables at the end of the context manager. """

    def __init__(self, modifies=None):
        self.modifies = modifies
        # the `no_annotations` context manager could be nested,
        # so we need a stack to keep track of the original states.
        self._orig_annotation_enabled = []

    def __enter__(self):
        global _annotation_enabled
        if self.modifies and len(self._orig_annotation_enabled) != 0:
            raise ValueError(
                "Cannot use `modifies` argument if `stop_annotating` is nested,"
                " e.g. if used as the `no_annotations` decorator.")
        self._orig_annotation_enabled.append(_annotation_enabled)
        _annotation_enabled = False

    def __exit__(self, *args):
        global _annotation_enabled
        _annotation_enabled = self._orig_annotation_enabled.pop()
        if self.modifies is not None:
            try:
                self.modifies.clear_block_variable()
            except AttributeError:
                for var in self.modifies:
                    var.clear_block_variable()


no_annotations = stop_annotating()
"""Decorator to turn off annotation for the decorated function."""


def annotate_tape(kwargs=None):
    """Return True if annotation flag is on, and False if not.

    If kwargs is given, the function will try to extract the
    annotate keyword. If the annotate keyword is not present it defaults to True.
    If annotation has been paused, then it will always return False.

    Args:
        kwargs (dict): A dictionary of keyword arguments to extract from.
            Note that this should be passed as a dictionary and not actual keyword arguments.

    Returns: bool

    """
    annotate = kwargs is None or kwargs.pop("annotate", True)

    # TODO: Consider if there is any scenario where one would want the keyword to have
    # precedence over the global flag.
    if not _annotation_enabled:
        return False

    return annotate


class Tape(object):
    """The tape.

    The tape consists of blocks, :class:`Block` instances.
    Each block represents one operation in the forward model.

    """
    __slots__ = ["_blocks", "_tf_tensors", "_tf_added_blocks", "_nodes",
                 "_tf_registered_blocks", "_bar", "_package_data",
                 "_checkpoint_manager", "latest_checkpoint",
                 "_eagerly_checkpoint_outputs", "_recompute_count"]

    def __init__(self, blocks=None, package_data=None):
        # Initialize the list of blocks on the tape.
        self._blocks = TimeStepSequence(blocks=blocks)
        # Dictionary of TensorFlow tensors. Key is id(block).
        self._tf_tensors = {}
        # Keep a list of blocks that has been added to the TensorFlow graph
        self._tf_added_blocks = []
        self._tf_registered_blocks = []
        self._bar = _NullProgressBar
        # Hook location for packages which need to store additional data on the
        # tape. Packages should store the data under a "packagename" key.
        self._package_data = package_data or {}
        # Default to checkpointing all block variables.
        self.latest_checkpoint = float("inf")
        self._checkpoint_manager = None
        # Whether to store the adjoint dependencies.
        self._eagerly_checkpoint_outputs = False
        # A counter for the number of tape recomputations.
        self._recompute_count = 0

    def clear_tape(self):
        """Clear the tape."""
        self.reset_variables()
        self._blocks = TimeStepSequence()
        for data in self._package_data.values():
            data.clear()
        self._checkpoint_manager = None
        self._recompute_count = 0

    @property
    def latest_timestep(self):
        """The current time step to which blocks will be added."""
        return max(len(self._blocks.steps) - 1, 0)

    @property
    def recompute_count(self):
        """The number of times the tape has been recomputed."""
        return self._recompute_count

    def end_timestep(self):
        """Mark the end of a timestep when taping the forward model."""
        if self._checkpoint_manager:
            self._checkpoint_manager.end_timestep(self.latest_timestep)
        else:
            self._blocks.append_step()

    def timestepper(self, iterable):
        """Return an iterator that advances the tape timestep.

        Note:
            This method facilitates taping timestepping simulations so that recompute
            checkpointing can be used on the tape. For example, a simulation with
            10 timesteps might use a timestepping loop of this form::

                tape = get_working_tape()

                for timestep in tape.timestepper(range(10)):
                    ...

            This has the effect of calling `tape.end_timestep()` after each iteration.

        Args:
            iterable (iterable): The iterable definining the sequence of timesteps.

        Returns:
            TapeTimeStepper: An iterator that advances the tape timestep.
        """
        return TapeTimeStepper(self, iterable)

    def reset_blocks(self):
        """Calls the Block.reset method of all blocks on the tape.

        Also resets any package-dependent data stored on the tape.
        """
        for block in self._blocks:
            block.reset()
        for data in self._package_data.values():
            data.reset()

    def add_block(self, block):
        """
        Adds a block to the tape and returns the index.
        """
        self._blocks.append(block)

        # len() is computed in constant time, so this should be fine.
        return len(self._blocks) - 1

    def add_to_checkpointable_state(self, block_var, last_used):
        """Add a block variable into the checkpointable state set.

        Note:
            `checkpointable_state` is a set of block variables which are needed
            to restart from the start of a timestep.

        Args:
            block_var (BlockVariable): The block variable to add.
            last_used (int): The last timestep in which the block variable was used.
        """
        if not self.timesteps:
            self._blocks.append_step()
        for step in self.timesteps[last_used + 1:]:
            step.checkpointable_state.add(block_var)

    def add_to_adjoint_dependencies(self, block_var, last_used):
        """Add a block variable into the adjoint dependencies set.

        Note:
            `adjoint_dependencies` is a set of block variables which are needed
            to compute the adjoint of a timestep.

        Args:
            block_var (BlockVariable): The block variable to add.
            last_used (int): The last timestep in which the block variable was used.
        """
        if not self.timesteps:
            self._blocks.append_step()
        for step in self.timesteps[last_used + 1:]:
            step.adjoint_dependencies.add(block_var)

    def enable_checkpointing(self, schedule, gc_timestep_frequency=None, gc_generation=2):
        """Enable checkpointing on the adjoint evaluation.

        A checkpoint manager able to execute the forward and adjoint computations
        according to the schedule provided by checkpoint_schedules package.

        Args:
            schedule (checkpoint_schedules.schedule): A schedule provided by the
            checkpoint_schedules package.
            gc_timestep_frequency (None or int): The timestep frequency for garbage collection.
            For additional information, please refer to the :class:`CheckpointManager`
            documentation.
            gc_generation (int): The generation for garbage collection. For additional
            information, please refer to the :class:`CheckpointManager` documentation.
        """
        if self._blocks:
            raise CheckpointError(
                "Checkpointing must be enabled before any blocks are added to the tape."
            )

        if gc_timestep_frequency is not None and not isinstance(gc_timestep_frequency, int):
            raise CheckpointError("gc_timestep_frequency must be an integer.")

        self._checkpoint_manager = CheckpointManager(
            schedule, self, gc_timestep_frequency=gc_timestep_frequency,
            gc_generation=gc_generation)

    def get_blocks(self, tag=None):
        """Returns a list of the blocks on the tape.

        Use the kwarg `tag` to optionally get all
        blocks with a specified tag.

        Returns:
            list[block.Block]: A list of :class:`Block` instances.

        """
        if tag is None:
            return self._blocks
        else:
            return [block for block in self._blocks if block.tag == tag]

    def get_tags(self):
        """
        Returns a list of the unique tags used in blocks on the tape.
        """
        tags = []
        for block in self._blocks:
            if block.tag is not None and block.tag not in tags:
                tags.append(block.tag)
        return tags

    def evaluate_adj(self, last_block=0, markings=False):
        """Evaluate the adjoint of the tape.

        Args:
            last_block (int, optional): The index of the last block to evaluate.
            markings (bool, optional): If True, then each `BlockVariable` of the current block
                will have set `is_control_dependent` attribute indicating whether their adjoint
                components are relevant for computing the final target adjoint values and
                `is_functional_dependency` attribute indicating whether its value impacts the
                value of the functional.
        """
        if self._checkpoint_manager:
            self._checkpoint_manager.evaluate_adj(last_block, markings)
        else:
            for i in self._bar("Evaluating adjoint").iter(
                range(len(self._blocks) - 1, last_block - 1, -1)
            ):
                self._blocks[i].evaluate_adj(markings=markings)

    def evaluate_tlm(self, markings=False):
        for i in self._bar("Evaluating TLM").iter(
            range(len(self._blocks))
        ):
            self._blocks[i].evaluate_tlm(markings=markings)

    def evaluate_hessian(self, markings=False):
        for i in self._bar("Evaluating Hessian").iter(
            range(len(self._blocks) - 1, -1, -1)
        ):
            self._blocks[i].evaluate_hessian(markings=markings)

    def reset_variables(self, types=None):
        for i in range(len(self._blocks) - 1, -1, -1):
            self._blocks[i].reset_variables(types)

    def reset_hessian_values(self):
        for i in range(len(self._blocks) - 1, -1, -1):
            self._blocks[i].reset_variables(types=("hessian"))

    def reset_tlm_values(self):
        for i in range(len(self._blocks) - 1, -1, -1):
            self._blocks[i].reset_variables(types=("tlm"))

    def copy(self):
        """Returns a shallow copy of the tape.

        Returns:
            Tape: The copy of the tape.

        """
        # TODO: Offer deepcopying. But is it feasible memory wise to copy all checkpoints?
        return Tape(
            blocks=self._blocks,
            package_data={k: v.copy() for k, v in self._package_data.items()}
        )

    def checkpoint_block_vars(self, controls=[], tag=None):
        """Returns an object to checkpoint the current state of all block variables on the tape.

        Args:
            controls (list): A list of controls for which the block variables should also be extracted.

        Returns:
            dict[BlockVariable, object]: The checkpointed block variables of the tape.

        """

        state_dict = {
            var: var.checkpoint
            for var in chain(
                chain.from_iterable(b.get_outputs() for b in self.get_blocks(tag)),
                (control.block_variable for control in controls))
        }
        state_dict["package_data"] = {k: v.checkpoint() for k, v in self._package_data.items()}
        return state_dict

    def restore_block_vars(self, block_vars):
        """Set the checkpoints of the tape according to a checkpoint dictionary.

        Args:
            block_vars (dict[BlockVariable, object]): A checkpoint object from checkpoint_block_vars.

        """
        block_vars = block_vars.copy()
        package_data = block_vars.pop("package_data")

        for k, v in block_vars.items():
            # we use the private _checkpoint attribute
            # here because the public attribute is a no-op
            # if the control values are "active", but we
            # need to make sure they are reset to the
            # cached value as well
            k._checkpoint = v

        for k, v in self._package_data.items():
            v.restore_from_checkpoint(package_data[k])

    def optimize(self, controls=None, functionals=None):
        if controls is not None:
            self.optimize_for_controls(controls)

        if functionals is not None:
            self.optimize_for_functionals(functionals)

    def optimize_for_controls(self, controls):
        # TODO: Consider if we want Enlist wherever it is possible. Like in this case.
        # TODO: Consider warning/message on empty tape.
        nodes = set([control.block_variable for control in controls])
        discarded_variables = set()
        optimized_timesteps = TimeStepSequence()

        for step in self._blocks.steps:
            optimized_timesteps.append_step()

            for block in step:
                depends_on_control = False
                for dep in block.get_dependencies():
                    if dep in nodes:
                        depends_on_control = True
                        break

                if depends_on_control:
                    for output in block.get_outputs():
                        if output in nodes:
                            raise RuntimeError("Control depends on another control.")
                        nodes.add(output)
                    optimized_timesteps.append(block)
                else:
                    discarded_variables.union(block.get_outputs())
            optimized_timesteps.steps[-1].checkpointable_state = \
                step.checkpointable_state - discarded_variables

        self._blocks = optimized_timesteps

    def optimize_for_functionals(self, functionals):
        retained_nodes = set([functional.block_variable
                             for functional in functionals]
                             )
        optimized_timesteps = []

        for step in reversed(self._blocks.steps):
            current_blocks = []
            for block in reversed(step):
                produces_functional = False
                for dep in block.get_outputs():
                    if dep in retained_nodes:
                        produces_functional = True

                if produces_functional:
                    for dep in block.get_dependencies():
                        retained_nodes.add(dep)
                    current_blocks.append(block)
            optimized_timesteps.append(TimeStep(reversed(current_blocks)))

        optimized_timesteps.reverse()

        for step, new_step in zip(self._blocks.steps, optimized_timesteps):
            new_step.checkpointable_state = \
                step.checkpointable_state & retained_nodes

        self._blocks = TimeStepSequence(steps=optimized_timesteps)

    @contextmanager
    def marked_control_dependents(self, controls):
        """Mark all the block variables on which the given controls depend."""
        nodes = self._nodes_dependent_on_controls(controls)
        for node in nodes:
            node.is_control_dependent = True
        try:
            yield
        finally:
            for node in nodes:
                node.is_control_dependent = False

    def _nodes_dependent_on_controls(self, controls):
        # This method is just a stripped down Block.optimize_for_controls
        nodes = set([control.block_variable for control in controls])

        for block in self.get_blocks():
            depends_on_control = False
            for dep in block.get_dependencies():
                if dep in nodes:
                    depends_on_control = True

            if depends_on_control:
                for output in block.get_outputs():
                    nodes.add(output)
        return nodes

    @contextmanager
    def marked_functional_dependencies(self, functional):
        """Mark all of the block variables on which the functional depends."""
        nodes = self._functional_dependencies(functional)
        for node in nodes:
            node.is_functional_dependency = True
        try:
            yield
        finally:
            for node in nodes:
                node.is_functional_dependency = False
 
    def _functional_dependencies(self, functional):
        # This function is just a stripped down Block.optimize_for_controls
        nodes = set([functional.block_variable])

        for block in reversed(self.get_blocks()):
            functional_dependency = False
            for output in block.get_outputs():
                if output in nodes:
                    functional_dependency = True

            if functional_dependency:
                for dep in block.get_dependencies():
                    nodes.add(dep)
        return nodes

    @property
    def timesteps(self):
        """Return the list of time steps on this tape."""
        return self._blocks.steps

    def _valid_tf_scope_name(self, name):
        """Return a valid TensorFlow scope name"""
        valid_name = ""
        p = re.compile("[A-Za-z0-9_.\\-]")
        for ch in name:
            match = p.match(ch)
            if not match:
                if valid_name and valid_name[-1] != "_":
                    valid_name += "_"
            else:
                valid_name += ch
        return valid_name

    def _get_tf_scope_name(self, block):
        """Return a TensorFlow scope name based on the block's class name."""
        # If the block is a BlockVariable we use the class name of block.output
        if block.__class__.__name__ == "BlockVariable":
            if block.output.__class__.__name__ in ("AdjFloat",):
                name = str(block.output.__class__.__name__) + "_" + str(block)
            else:
                name = str(block.output.__class__.__name__)
        else:
            name = block.__class__.__name__
        return self._valid_tf_scope_name(name)

    def _tf_register_blocks(self, name=None):
        lst = list()
        lst.append(name)
        for block in self.get_blocks():
            if block in self._tf_added_blocks:
                continue
            self._tf_added_blocks.append(block)
            lst.append(block)
        self._tf_registered_blocks.append(lst)

    def _tf_rebuild_registered_blocks(self):
        """Remove blocks that no longer exist on the tape from registered blocks."""
        new_registered_blocks = []
        new_added_blocks = []
        for scope in self._tf_registered_blocks:
            lst = [scope[0]]
            for i in range(1, len(scope)):
                block = scope[i]
                if block in self.get_blocks():
                    lst.append(block)
                    new_added_blocks.append(block)

            if len(lst) > 1:
                new_registered_blocks.append(lst)
        self._tf_registered_blocks = new_registered_blocks
        self._tf_added_blocks = new_added_blocks

    def _tf_add_blocks(self):
        """Add new blocks to the TensorFlow graph."""

        import tensorflow as tf

        self._tf_register_blocks()
        self._tf_rebuild_registered_blocks()

        for scope in self._tf_registered_blocks:
            scope_name = scope[0]
            with tf.name_scope(scope_name):
                for i in range(1, len(scope)):
                    block = scope[i]

                    # Block dependencies
                    in_tensors = []
                    for dep in block.get_dependencies():
                        if id(dep) in self._tf_tensors:
                            in_tensors.append(self._tf_tensors[id(dep)])
                        else:
                            with tf.name_scope(self._get_tf_scope_name(dep)):
                                tin = tf.py_func(lambda: None, [], [tf.float64],
                                                 name=self._valid_tf_scope_name(str(dep)))
                                in_tensors.append(tin)
                                self._tf_tensors[id(dep)] = tin

                    # Block node
                    with tf.name_scope(self._get_tf_scope_name(block)):
                        tensor = tf.py_func(lambda: None, in_tensors, [tf.float64],
                                            name=self._valid_tf_scope_name(str(block)))
                        self._tf_tensors[id(block)] = tensor

                    # Block outputs
                    for out in block.get_outputs():
                        with tf.name_scope(self._get_tf_scope_name(out)):
                            tout = tf.py_func(lambda: None, [tensor], [tf.float64],
                                              name=self._valid_tf_scope_name(str(out)))
                            self._tf_tensors[id(out)] = tout

    @contextmanager
    def name_scope(self, name=None):
        """Returns a context manager that creates hierarchical names for TensorFlow operations.

        Args:
            name (str|None): Name of scope to use. Default None.
        """
        self._tf_register_blocks()
        yield
        self._tf_register_blocks(name)

    def visualise(self, output="log", launch_tensorboard=False, open_in_browser=False):
        """Makes a visualisation of the tape as a graph using TensorFlow
        or GraphViz. (Default: Tensorflow). If `output` endswith `.dot` or
        `.pdf`, Graphviz is used.

        Args:
            output (str): Directory where event files for TensorBoard is
                stored. Default log.
            launch_tensorboard (bool): Launch TensorBoard in the background.
                Default False.
            open_in_browser (bool): Opens http://localhost:6006/ in a web
                browser. Default False.
        """
        if output.endswith(".dot"):
            return self.visualise_dot(output)
        elif output.endswith(".pdf"):
            return self.visualise_pdf(output)

        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        self._tf_add_blocks()

        # Write graph to file
        with tf.compat.v1.Session() as sess:
            writer = tf.compat.v1.summary.FileWriter(output, sess.graph)
            writer.close()

        if not launch_tensorboard or not open_in_browser:
            print("Run the command line:\n"
                  "--> tensorboard --logdir={}\n"
                  "Then open http://localhost:6006/ in your web browser.".format(output))

        if launch_tensorboard:
            def launchTensorBoard():
                os.system('tensorboard --logdir=' + output)

            t = threading.Thread(target=launchTensorBoard, args=([]))
            t.start()

        if open_in_browser:
            import webbrowser
            webbrowser.open_new_tab("http://localhost:6006/")

    def create_graph(self, backend="networkx"):
        import networkx as nx
        G = nx.DiGraph()
        for i, block in enumerate(self._blocks):
            block.create_graph(G, pos=i)
        return G

    def visualise_dot(self, filename):
        """Makes a visualisation of the tape in dot format and you
        can render it using Graphviz dot.

        Args:
            filename (str): File to save the visualisation. Default None.
        """
        G = self.create_graph()
        from networkx.drawing.nx_agraph import write_dot
        write_dot(G, filename)

    def visualise_pdf(self, filename):
        """Create a PDF visualisation of the tape.

        This depends on the Python package networkx and the external Graphviz
        package. The latter can be installed using e.g.::

            sudo apt install graphviz

        on Ubuntu or::

            brew install graphviz

        on Mac.

        Args:
            filename (str): File to save the visualisation. Must end in .pdf.
        """
        if not filename.endswith(".pdf"):
            raise ValueError("Filename for PDF output must end in .pdf")
        from networkx.drawing.nx_agraph import to_agraph
        A = to_agraph(self.create_graph())
        A.draw(filename, prog="dot")

    @property
    def progress_bar(self):
        """Specify a progress bar class to print during tape evaluation.

        Setting this attribute to a subclass of :class:`progress.bar.Bar` will
        cause every evaluation of a reduced functional, adjoint, TLM or Hessian
        to print a progress bar.

        For example, the following code::

            from progress.bar import FillingSquaresBar
            tape = get_working_tape()
            tape.progress_bar = FillingSquaresBar

        will cause tape evaluations to print progress bars similar to the
        following::

            Evaluating functional ▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣ 100%
            Evaluating adjoint ▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣ 100%

        For information on available progress bar styles and their
        configuration, see the `progress package documentation
        <https://pypi.org/project/progress/>`_.
        """
        return self._bar

    @progress_bar.setter
    def progress_bar(self, bar):
        self._bar = bar


class _NullProgressBar:
    """A placeholder class with the same interface as a progress bar."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass

    def iter(self, iterator):
        return iterator


class TapeTimeStepper:
    """Iterator wrapper which advances the timestep after each iteration."""
    def __init__(self, tape, iterable):
        self.tape = tape
        self.iterator = tape.progress_bar("Taping forward").iter(iterable)
        self._first = True

    def __iter__(self):
        return self

    def __next__(self):
        step = next(self.iterator)
        if self._first:
            self._first = False
        else:
            self.tape.end_timestep()
        return step


class TimeStep(list):
    """A list of blocks in a single time step, plus associated metadata."""
    def __init__(self, blocks=()):
        super().__init__(blocks)
        # The set of block variables which are needed to restart from the start
        # of this timestep.
        self.checkpointable_state = OrderedSet()
        self.adjoint_dependencies = OrderedSet()
        # A dictionary mapping the block variables in the checkpointable state
        # to their checkpoint values.
        self._checkpoint = {}
        # Flag indicating whether the adjoint dependencies have been revised
        # by removing outputs not marked in the path and adding checkpointable
        # states that are marked in the path.
        self._revised_adj_deps = False

    def copy(self, blocks=None):
        out = TimeStep(blocks or self)
        out.checkpointable_state = self.checkpointable_state
        return out

    def checkpoint(self, checkpointable_state, adj_dependencies, global_deps):
        """Store a copy of the checkpoints in the checkpointable state.

        Args:
            checkpointable_state (bool): If True, store the checkpointable state
            required to restart from the start of a timestep.
            adj_dependencies (bool): If True, store the adjoint dependencies required
            to compute the adjoint of a timestep.
            global_deps (set): This set stores the common dependencies for all timesteps.
            For additional information, please refer to the :class:`CheckpointManager`
            documentation.
        """
        with stop_annotating():
            if checkpointable_state:
                for var in self.checkpointable_state:
                    if var in global_deps:
                        # Creating a new checkpoint object is not necessary here
                        # because the global dependencies do not change.
                        self._checkpoint[var] = var._checkpoint
                    else:
                        self._checkpoint[var] = var.saved_output._ad_create_checkpoint()

            if adj_dependencies:
                if self._revised_adj_deps:
                    for var in self.adjoint_dependencies:
                        self._checkpoint[var] = var.saved_output._ad_create_checkpoint()
                else:
                    # The adjoint dependencies have not been revised yet. At this stage,
                    # the block nodes are not marked in the path because the control variable(s)
                    # are not yet determined.
                    for var in self.adjoint_dependencies.union(self.checkpointable_state):
                        self._checkpoint[var] = var.saved_output._ad_create_checkpoint()

    def restore_from_checkpoint(self, from_storage):
        """Restore the block var checkpoints from the timestep checkpoint."""
        from .overloaded_type import OverloadedType
        for var, checkpoint in self._checkpoint.items():
            if (
                from_storage == StorageType.DISK
                and isinstance(checkpoint, OverloadedType)
            ):
                # checkpoint._ad_restore_checkpoint should be able to restore
                # from disk.
                var.checkpoint = checkpoint._ad_restore_at_checkpoint(checkpoint)
            else:
                var.checkpoint = checkpoint

    def delete_checkpoint(self):
        """Delete the stored checkpoint references."""
        self._checkpoint = {}


class TimeStepSequence(list):
    """A list of Blocks separated into timesteps to facilitate checkpointing.

    This behaves like a list of blocks. To access a list of the timesteps, use
    the :attr:`steps` property.

    Args:
        blocks (list[Block] or TimeStepSequence, optional): If provided, `blocks` can be a list of :class:`Block`
        or another TimeStepSequence to copy from.
        steps (list[TimeStep], optional): If provided, `steps` should be a list of :class:`TimeStep` to copy from.

    Attributes:
        steps (list[TimeStep]): A list of timesteps.
    """

    def __init__(self, blocks=None, steps: Optional[Iterable[Iterable[TimeStep]]] = None):
        # Keep both per-timestep and unified block lists.
        if steps and blocks:
            raise ValueError("set blocks or steps but not both.")
        elif isinstance(blocks, TimeStepSequence):
            self._steps = [step.copy() for step in blocks._steps]
        elif blocks:
            self._steps = [TimeStep(blocks)]
        else:
            self._steps = list(step.copy() for step in steps) if steps else []
        super().__init__(chain.from_iterable(self._steps))

    @property
    def steps(self):
        return self._steps

    def append(self, other):
        """Add a new block to the sequence and to the current TimeStep."""
        if not self.steps:
            self.append_step()
        self._steps[-1].append(other)
        super().append(other)

    def append_step(self, step=None):
        """Add a new TimeStep."""
        self._steps.append(step or TimeStep())

    def __setitem__(self, key, value):
        raise ValueError(
            "Unable to set arbitrary blocks. Try appending instead."
        )

    def __delitem__(self, key, value):
        raise ValueError(
            "Unable to delete blocks from sequence."
        )


class TapePackageData(ABC):
    """Abstract base class for additional data that packages store on the tape.

    If a package that uses Pyadjoint needs to store additional tape state, such
    as the location of checkpoint files, it should store an instance of a
    subclass of `TapePackageData` in the :attr:`_package_data` dictionary of
    the tape. E.g::

        get_working_tape()._package_data["firedrake"] = checkpoint_data
    """

    @abstractmethod
    def clear(self):
        """Delete the current state of the object.

        This is called when the tape is cleared."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the current state before a new forward evaluation."""
        pass

    @abstractmethod
    def checkpoint(self):
        """Record the information required to return to the current state."""
        pass

    @abstractmethod
    def restore_from_checkpoint(self, state):
        """Restore state from a previously stored checkpoint."""
        pass

    @abstractmethod
    def copy(self):
        """Produce a new copy of state to be passed to a copy of the tape."""
        pass

    @abstractmethod
    def continue_checkpointing(self):
        """Continue the checkpointing process on disk.
        """
        pass

    @abstractmethod
    def pause_checkpointing(self):
        """Pause the checkpointing process on disk.
        """
        pass
