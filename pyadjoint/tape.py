import backend


_working_tape = None


def get_working_tape():
    return _working_tape


def set_working_tape(tape):
    global _working_tape
    _working_tape = tape


class Tape(object):
    """The tape.

    The tape consists of blocks, :class:`Block` instances.
    Each block represents one operation in the forward model.

    TODO: Does the name "Tape" even make sense in this case?
    That is the tape should be what binds blocks together?
    Maybe even BlockOutputs are tapes? Might be worth considering,
    as I'm not particularly satisfied with the name "BlockOutput".
    However ultimately the blocks themselves reference the BlockOutput,
    and not the other way around. So they don't actually function as
    one would think of a Tape.

    """
    __slots__ = ["_blocks"]

    def __init__(self):
        # Initialize the list of blocks on the tape.
        self._blocks = []

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

    def evaluate(self):
        for i in range(len(self._blocks)-1, -1, -1):
            self._blocks[i].evaluate_adj()

    def reset_variables(self):
        for i in range(len(self._blocks)-1, -1, -1):
            self._blocks[i].reset_variables()

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
            filename (:obj:`str`, optional): File to save the visualisation. Default None.
            scale (:obj:`float`, optional): Scales the distances between nodes.
                Only relevant for dot set to False. Default 1.0.
            dot (:obj:`bool`, optional): Write to specified file in dot-format. Default False.

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


            node_colors = nx.get_node_attributes(G, 'node_color').values()
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


class Block(object):
    """Base class for all Tape Block types.
    
    Each instance of a Block type represents an elementary operation in the
    forward model.
    
    Abstract methods
        :func:`evaluate_adj`

    """
    __slots__ = ['_dependencies', '_outputs']

    def __init__(self):
        self._dependencies = []
        self._outputs = []

    def add_dependency(self, dep):
        """Adds object to the block dependencies if it has not already been added.

        Args:
            dep (:class:`BlockOutput`): The object to be added.

        """
        if not dep in self._dependencies: # Can be optimized if we have need for huge lists.
            self._dependencies.append(dep)

    def get_dependencies(self):
        """Returns the list of dependencies.

        Returns:
            :obj:`list`: A list of :class:`BlockOutput` instances.

        """
        return self._dependencies

    def add_output(self, obj):
        """Adds object to the block output list.

        Args:
            obj (:class:`BlockOutput`): The object to be added.

        """
        self._outputs.append(obj)

    def get_outputs(self):
        """Returns the list of block outputs.

        Returns:
            :obj:`list`: A list of :class:`BlockOutput` instances.

        """
        return self._outputs

    def reset_variables(self):
        """Resets all adjoint variables in the block dependencies.

        """
        for dep in self._dependencies:
            dep.reset_variables()

    def evaluate_adj(self):
        """This method must be overriden.
        
        The method should implement a routine for evaluating the adjoint of the block.

        """
        raise NotImplementedError

    def create_graph(self, G, pos, scale=1.0):

        # Edges for block dependencies
        for xpos, dep in enumerate(self.get_dependencies()):
            G.add_edge(id(dep), id(self))
            if "label" not in G.node[id(dep)]:
                G.node[id(dep)]['label'] = str(dep)
                G.node[id(dep)]['node_color'] = "r"
                G.node[id(dep)]['position'] = (scale*(0.1*xpos), scale*(-pos+0.5))

        # Edges for block outputs
        for xpos, out in enumerate(self.get_outputs()):
            G.add_edge(id(self), id(out))
            if "label" not in G.node[id(out)]:
                G.node[id(out)]['label'] = str(out)
                G.node[id(out)]['node_color'] = "r"
                G.node[id(out)]['position'] = (scale*(0.1*xpos), scale*(-pos-0.5))

        # Set properties for Block node
        G.node[id(self)]['label'] = str(self)
        G.node[id(self)]['node_color'] = "b"
        G.node[id(self)]['position'] = (0, scale*(-pos))
        G.node[id(self)]['shape'] = "box"


class BlockOutput(object):
    """References a block output variable.

    """
    id_cnt = 0

    def __init__(self, output):
        self.output = output
        self.adj_value = 0
        self.saved_output = None
        BlockOutput.id_cnt += 1
        self.id = BlockOutput.id_cnt

    def add_adj_output(self, val):
        self.adj_value += val

    def get_adj_output(self):
        #print "Bugger ut: ", self.adj_value
        #print self.output
        return self.adj_value

    def set_initial_adj_input(self, value):
        self.adj_value = value

    def reset_variables(self):
        self.adj_value = 0

    def get_output(self):
        return self.output

    def save_output(self):
        # Previously I used 
        # self.saved_ouput = Function(self.output.function_space(), self.output.vector()) as
        # assign allocates a new vector (and promptly doesn't need nor 
        # modifies the old vector) However this does not work when we also want to save copies of
        # other functions, say an output function from a SolveBlock. As
        # backend.solve overwrites the vector of the solution function.

        self.saved_output = self.output.copy(deepcopy=True)

    def get_saved_output(self):
        if self.saved_output:
            return self.saved_output
        else:
            return self.output

    def __str__(self):
        return str(self.output)


class OverloadedType(object):
    """Base class for OverloadedType types.

    The purpose of each OverloadedType is to extend a type such that
    it can be referenced by blocks as well as overload basic mathematical
    operations such as __mul__, __add__, where they are needed.

    """
    def __init__(self, *args, **kwargs):
        tape = kwargs.pop("tape", None)

        if tape:
            self.tape = tape
        else:
            self.tape = get_working_tape()

        self.original_block_output = self.create_block_output()

    def create_block_output(self):
        block_output = BlockOutput(self)
        self.set_block_output(block_output)
        return block_output

    def set_block_output(self, block_output):
        self.block_output = block_output

    def get_block_output(self):
        return self.block_output

    def get_adj_output(self):
        return self.original_block_output.get_adj_output()

    def set_initial_adj_input(self, value):
        self.block_output.set_initial_adj_input(value)

    def reset_variables(self):
        self.original_block_output.reset_variables()


