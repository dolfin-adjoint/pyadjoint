# Type dependencies
import os
import re
import threading
from contextlib import contextmanager
from functools import wraps
from itertools import chain
from abc import ABC, abstractmethod


_working_tape = None
_annotation_enabled = False


def get_working_tape():
    return _working_tape


def pause_annotation():
    global _annotation_enabled
    _annotation_enabled = False


def continue_annotation():
    global _annotation_enabled
    _annotation_enabled = True
    return _annotation_enabled


class set_working_tape(object):
    """A context manager whithin which a new tape is set as the working tape.
       This context manager can also be used in an imperative manner.

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


class stop_annotating(object):
    """A context manager within which annotation is stopped.

    Args:
        modifies (OverloadedType or list[OverloadedType]): One or more
            variables which appear in the tape and whose values are to be
            changed inside the context manager.

    The `modifies` argument is intended to be used by user code which
    changes the value of inputs to the adjoint calculation such as time varying
    forcings. Its effect is to create a new block variable for each of the
    modified variables at the end of the context manager. """

    def __init__(self, modifies=None):
        global _annotation_enabled
        self.modifies = modifies
        self._orig_annotation_enabled = _annotation_enabled

    def __enter__(self):
        global _annotation_enabled
        _annotation_enabled = False

    def __exit__(self, *args):
        global _annotation_enabled
        _annotation_enabled = self._orig_annotation_enabled
        if self.modifies is not None:
            try:
                self.modifies.create_block_variable()
            except AttributeError:
                for var in self.modifies:
                    var.create_block_variable()


def no_annotations(function):
    """Decorator to turn off annotation for the decorated function."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with stop_annotating():
            return function(*args, **kwargs)

    return wrapper


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


def _find_relevant_nodes(tape, controls):
    # This function is just a stripped down Block.optimize_for_controls
    blocks = tape.get_blocks()
    nodes = set([control.block_variable for control in controls])

    for block in blocks:
        depends_on_control = False
        for dep in block.get_dependencies():
            if dep in nodes:
                depends_on_control = True

        if depends_on_control:
            for output in block.get_outputs():
                nodes.add(output)
    return nodes


class Tape(object):
    """The tape.

    The tape consists of blocks, :class:`Block` instances.
    Each block represents one operation in the forward model.

    """
    __slots__ = ["_blocks", "_tf_tensors", "_tf_added_blocks", "_nodes",
                 "_tf_registered_blocks", "_bar", "_package_data"]

    def __init__(self, blocks=None, package_data=None):
        # Initialize the list of blocks on the tape.
        self._blocks = [] if blocks is None else blocks
        # Dictionary of TensorFlow tensors. Key is id(block).
        self._tf_tensors = {}
        # Keep a list of blocks that has been added to the TensorFlow graph
        self._tf_added_blocks = []
        self._tf_registered_blocks = []
        self._bar = _NullProgressBar
        # Hook location for packages which need to store additional data on the
        # tape. Packages should store the data under a "packagename" key.
        self._package_data = package_data or {}

    def clear_tape(self):
        self.reset_variables()
        self._blocks = []
        for data in self._package_data.values():
            data.clear()

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
        for i in self._bar("Evaluating adjoint").iter(
            range(len(self._blocks) - 1, last_block - 1, -1)
        ):
            self._blocks[i].evaluate_adj(markings=markings)

    def evaluate_tlm(self):
        for i in self._bar("Evaluating TLM").iter(
            range(len(self._blocks))
        ):
            self._blocks[i].evaluate_tlm()

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
            blocks=self._blocks[:],
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
        blocks = self.get_blocks()
        nodes = set([control.block_variable for control in controls])
        valid_blocks = []

        for block in blocks:
            depends_on_control = False
            for dep in block.get_dependencies():
                if dep in nodes:
                    depends_on_control = True

            if depends_on_control:
                for output in block.get_outputs():
                    if output in nodes:
                        raise RuntimeError("Control depends on another control.")
                    nodes.add(output)
                valid_blocks.append(block)
        self._blocks = valid_blocks

    def optimize_for_functionals(self, functionals):
        blocks = self.get_blocks()
        nodes = set([functional.block_variable for functional in functionals])
        valid_blocks = []

        for block in reversed(blocks):
            produces_functional = False
            for dep in block.get_outputs():
                if dep in nodes:
                    produces_functional = True

            if produces_functional:
                for dep in block.get_dependencies():
                    nodes.add(dep)
                valid_blocks.append(block)
        self._blocks = list(reversed(valid_blocks))

    @contextmanager
    def marked_nodes(self, controls):
        nodes = _find_relevant_nodes(self, controls)
        for node in nodes:
            node.marked_in_path = True
        yield
        for node in nodes:
            node.marked_in_path = False

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

    def __exit__(self):
        pass

    def iter(self, iterator):
        return iterator


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
        """Restore state from a previously stored checkpioint."""
        pass

    @abstractmethod
    def copy(self):
        """Produce a new copy of state to be passed to a copy of the tape."""
        pass
