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

    def create_graph(self):
        import networkx as nx

        G = nx.Graph()
        for block in self._blocks:
            block.create_graph(G)

        return G


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

    def create_graph(self, G):
        deps = get_dependencies()
        outs = get_outputs()

        G.add_edge(deps.get_name(), outs.get_name())

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

    def get_name(self):
        return self.id


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


