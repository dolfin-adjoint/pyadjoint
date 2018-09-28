# Type dependencies
from . import block
import re
import os
import threading
from contextlib import contextmanager

# TOOD: Save/checkpoint functions always. Not just on assign.

_working_tape = None
_stop_annotating = 0


def get_working_tape():
    return _working_tape


def set_working_tape(tape):
    global _working_tape
    _working_tape = tape


def pause_annotation():
    global _stop_annotating
    _stop_annotating += 1


def continue_annotation():
    global _stop_annotating
    _stop_annotating -= 1
    return _stop_annotating <= 0


class stop_annotating(object):
    def __enter__(self):
        pause_annotation()

    def __exit__(self, *args):
        continue_annotation()

def no_annotations(function):
    """Decorator to turn off annotation for the decorated function."""
    def wrapper(*args, **kwargs):
        with stop_annotating():
            return function(*args, **kwargs)
    return wrapper


def annotate_tape(kwargs=None):
    """Returns True if annotation flag is on, and False if not.

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
    if _stop_annotating > 0:
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
    __slots__ = ["_blocks", "_tf_tensors", "_tf_added_blocks", "_nodes", "_tf_registered_blocks"]

    def __init__(self, blocks=None):
        # Initialize the list of blocks on the tape.
        self._blocks = [] if blocks is None else blocks
        # Dictionary of TensorFlow tensors. Key is id(block).
        self._tf_tensors = {}
        # Keep a list of blocks that has been added to the TensorFlow graph
        self._tf_added_blocks = []
        self._tf_registered_blocks = []

    def clear_tape(self):
        self.reset_variables()
        self._blocks = []

    def add_block(self, block):
        """
        Adds a block to the tape and returns the index.
        """
        self._blocks.append(block)

        # len() is computed in constant time, so this should be fine.
        return len(self._blocks)-1

    def get_blocks(self):
        """Returns a list of the blocks on the tape.

        Returns:
            list[block.Block]: A list of :class:`Block` instances.

        """
        return self._blocks

    def evaluate_adj(self, last_block=0, markings=False):
        for i in range(len(self._blocks)-1, last_block-1, -1):
            self._blocks[i].evaluate_adj(markings=markings)

    def evaluate_tlm(self):
        for i in range(len(self._blocks)):
            self._blocks[i].evaluate_tlm()

    def evaluate_hessian(self, markings=False):
        for i in range(len(self._blocks)-1, -1, -1):
            self._blocks[i].evaluate_hessian(markings=markings)

    def reset_variables(self, types=None):
        for i in range(len(self._blocks)-1, -1, -1):
            self._blocks[i].reset_variables(types)

    def reset_hessian_values(self):
        for i in range(len(self._blocks)-1, -1, -1):
            self._blocks[i].reset_variables(types=("hessian"))

    def reset_tlm_values(self):
        for i in range(len(self._blocks)-1, -1, -1):
            self._blocks[i].reset_variables(types=("tlm"))

    def copy(self):
        """Returns a shallow copy of the tape.

        Returns:
            Tape: The copy of the tape.

        """
        # TODO: Offer deepcopying. But is it feasible memory wise to copy all checkpoints?
        return Tape(blocks=self._blocks[:])

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
        l = []
        l.append(name)
        for block in self.get_blocks():
            if block in self._tf_added_blocks:
                continue
            self._tf_added_blocks.append(block)
            l.append(block)
        self._tf_registered_blocks.append(l)

    def _tf_rebuild_registered_blocks(self):
        """Remove blocks that no longer exist on the tape from registered blocks."""
        new_registered_blocks = []
        new_added_blocks = []
        for scope in self._tf_registered_blocks:
            l = [scope[0]]
            for i in range(1, len(scope)):
                block = scope[i]
                if block in self.get_blocks():
                    l.append(block)
                    new_added_blocks.append(block)

            if len(l) > 1:
                new_registered_blocks.append(l)
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

    def visualise(self, logdir="log", launch_tensorboard=False, open_in_browser=False):
        """Makes a visualisation of the tape as a graph using TensorFlow.

        Args:
            logdir (str): Directory where event files for TensorBoard is stored. Default log.
            launch_tensorboard (bool): Launch TensorBoard in the background. Default False.
            open_in_browser (bool): Opens http://localhost:6006/ in a web browser. Default False.
        """

        import tensorflow as tf
        tf.reset_default_graph()
        self._tf_add_blocks()

        # Write graph to file
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir, sess.graph)
            writer.close()

        if not launch_tensorboard or not open_in_browser:
            print("Run the command line:\n" \
                  "--> tensorboard --logdir={}\n" \
                  "Then open http://localhost:6006/ in your web browser.".format(logdir))

        if launch_tensorboard:
            def launchTensorBoard():
                os.system('tensorboard --logdir=' + logdir)

            t = threading.Thread(target=launchTensorBoard, args=([]))
            t.start()

        if open_in_browser:
            import webbrowser
            webbrowser.open_new_tab("http://localhost:6006/")
