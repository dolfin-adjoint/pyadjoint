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
        self.name = id_cnt

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
        return self.name


def create_overloaded_object(obj):
    """Creates an :class:`OverloadedType` instance corresponding `obj`.

    Args:
        obj: The object to create an overloaded object from.

    Returns:
        :class:`OverloadedType`: An object which has the same attributes as `obj`, but also the extra attributes/methods needed for use in the tape.

    Raises:
        NotImplemntedError: If the corresponding :class:`OverloadedType` has not been implemented.

    """
    if isinstance(obj, float):
        return AdjFloat(obj)
    elif isinstance(obj, backend.Function):
        # This will invoke the backend constructor in a way that is said to be only intended for internal library use. 

        return Function(obj)
    else:
        raise NotImplementedError


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


class Function(OverloadedType, backend.Function):
    def __init__(self, *args, **kwargs):
        super(Function, self).__init__(*args, **kwargs)
        backend.Function.__init__(self, *args, **kwargs)

    def assign(self, other, *args, **kwargs):
        annotate_tape = kwargs.pop("annotate_tape", True)
        if annotate_tape:
            block = AssignBlock(self, other)
            tape = get_working_tape()
            tape.add_block(block)
        
        #self.get_block_output().save_output()
        #self.set_block_output(other.get_block_output())
        #self.get_block_output().output = self

        return super(Function, self).assign(other, *args, **kwargs)


class AssignBlock(Block):
    def __init__(self, func, other):
        super(AssignBlock, self).__init__()
        self.add_dependency(func.get_block_output())
        self.add_dependency(other.get_block_output())
        func.get_block_output().save_output()
        other.get_block_output().save_output()


        self.add_output(func.create_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        
        self.get_dependencies()[1].add_adj_output(adj_input)


class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)


class DirichletBC(OverloadedType, backend.DirichletBC):
    def __init__(self, *args, **kwargs):
        super(DirichletBC, self).__init__(*args, **kwargs)
        backend.DirichletBC.__init__(self, *args, **kwargs)

class OverloadedExpressionMetaClass(backend.ExpressionMetaClass):
    def __new__(mcs, class_name, bases, dict_):
        # Some logic to exclude overloaded Expression from bases, and replace with backend.Expression.

        original = backend.ExpressionMetaClass.__new__(mcs, class_name, bases, dict_)

        # Some logic to wrap new compiledExpression into an overloaded CompiledExpression, and return it.

        pass

class CompiledExpression(OverloadedType):
    pass    


class Expression(OverloadedType, backend.Expression):
    __metaclass__ = OverloadedExpressionMetaClass
    def __new__(cls, cppcode=None, *args, **kwargs):
        original = backend.Expression.__new__(cls, cppcode, *args, **kwargs)
        return CompiledExpression(original)

    def __init__(self, *args, **kwargs):
        super(Expression, self).__init__(*args, **kwargs)
        backend.Expression.__init__(self, *args, **kwargs)


class AdjFloat(OverloadedType, float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, *args)

    def __init__(self, *args, **kwargs):
        super(AdjFloat, self).__init__(*args, **kwargs)
        float.__init__(self, *args, **kwargs)

    def __mul__(self, other):
        output = float.__mul__(self, other)
        if output is NotImplemented:
            return NotImplemented

        block = MulBlock(self, other)

        tape = get_working_tape()
        tape.add_block(block)

        output = create_overloaded_object(output)
        block.add_output(output.get_block_output())
        
        return output 


class MulBlock(Block):
    def __init__(self, lfactor, rfactor):
        super(MulBlock, self).__init__()
        self.lfactor = lfactor
        self.rfactor = rfactor

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()

        self.rfactor.add_adj_output(adj_input * self.lfactor)
        self.lfactor.add_adj_output(adj_input * self.rfactor)