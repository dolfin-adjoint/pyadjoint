# Type dependencies
from . import block

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


class Tape(object):
    """The tape.

    The tape consists of blocks, :class:`Block` instances.
    Each block represents one operation in the forward model.

    """
    __slots__ = ["_blocks"]

    def __init__(self, blocks=None):
        # Initialize the list of blocks on the tape.
        self._blocks = [] if blocks is None else blocks

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

    def evaluate(self, last_block=0):
        for i in range(len(self._blocks)-1, last_block-1, -1):
            self._blocks[i].evaluate_adj()

    def evaluate_tlm(self):
        for i in range(len(self._blocks)):
            self._blocks[i].evaluate_tlm()

    def evaluate_hessian(self):
        for i in range(len(self._blocks)-1, -1, -1):
            self._blocks[i].evaluate_hessian()

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

    def create_graph(self, backend="networkx", scale=1.0):
        import networkx as nx

        G = nx.DiGraph()
        for i, block in enumerate(self._blocks):
            block.create_graph(G, pos=i, scale=scale)

        return G

    def visualise(self, filename=None, scale=1.0, dot=False):
        """Makes a visualisation of the tape as a graph.

        For bigger tapes it is recommended to set the keyword argument
        `dot` to True. It should then save a file in dot format and you
        can render it using Graphviz dot.

        Args:
            filename (str|None): File to save the visualisation. Default None.
            scale (float): Scales the distances between nodes.
                Only relevant for dot set to False. Default 1.0.
            dot (bool): Write to specified file in dot-format. Default False.
                If this is True, then filename must be set.

        Raises:
            NotImplementedError: If you choose dot-format but supply no filename.

        """
        G = self.create_graph(scale=scale)

        if dot:
            from networkx.drawing.nx_agraph import write_dot
            if filename:
                write_dot(G, filename)
            else:
                raise NotImplementedError
        else:
            import networkx as nx
            import pylab as plt

            # Draw nodes
            fixed_node_positions = nx.get_node_attributes(G, 'position')
            pos = nx.spring_layout(G, pos=fixed_node_positions, fixed=fixed_node_positions.keys())


            node_colors = list(nx.get_node_attributes(G, 'node_color').values())
            nx.draw_networkx_nodes(G, pos,
                                   node_color=node_colors,
                                   node_size=500,
                                   alpha=0.8)
            node_labels = nx.get_node_attributes(G, 'label')

            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(G, pos, labels=node_labels)

            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)

            # Turn axis off
            plt.axis('off')

            # Show or save graph
            if not filename:
                plt.show()
                plt.clf()
            else:
                plt.savefig(filename)

