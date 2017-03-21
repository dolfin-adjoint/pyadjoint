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

    def visualise(self, filename=None, scale=1.0):
        import networkx as nx
        import matplotlib.pyplot as plt
 
        G = self.create_graph(scale=scale)
        
        # Draw nodes
        fixed_node_positions = nx.get_node_attributes(G, 'position')
        pos = nx.spring_layout(G, pos=fixed_node_positions, fixed=fixed_node_positions.keys())
       

        node_colors = nx.get_node_attributes(G, 'node_color').values()
        nx.draw_networkx_nodes(G, pos,
                               node_color=node_colors,
                               node_size=500,
                               alpha=0.8)
        node_labels = nx.get_node_attributes(G, 'state')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, labels=node_labels)

        edge_labels = nx.get_edge_attributes(G, 'state')
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
            G.add_edge(str(dep), str(self))
            G.edge[str(dep)][str(self)]['state'] = "dep"
            G.node[str(dep)]['state'] = str(dep)
            G.node[str(dep)]['node_color'] = "r"
            G.node[str(dep)]['position'] = (scale*(0.1*xpos), scale*(-pos+0.5))

        # Edges for block outputs
        for xpos, out in enumerate(self.get_outputs()):
            G.add_edge(str(self), str(out))
            G.edge[str(self)][str(out)]['state'] = "out"
            G.node[str(out)]['state'] = str(out)
            G.node[str(out)]['node_color'] = "r"
            G.node[str(out)]['position'] = (scale*(0.1*xpos), scale*(-pos-0.5))

        # Set properties for Block node
        G.node[str(self)]['state'] = str(self)
        G.node[str(self)]['node_color'] = "b"
        G.node[str(self)]['position'] = (0, scale*(-pos))


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

        return Function(obj.function_space(), obj.vector())
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


_backend_ExpressionMetaClass = backend.functions.expression.ExpressionMetaClass
class OverloadedExpressionMetaClass(_backend_ExpressionMetaClass):
    """Overloads the ExpressionMetaClass so that we can create overloaded 
    Expression types.

    """
    def __new__(mcs, class_name, bases, dict_):
        """Handles creation of new Expression classes.

        """
        if class_name == "Expression" or class_name == "CompiledExpression":
            if len(bases) >= 1 and bases[0] == OverloadedType:
                # If this is our own overloaded Expression/CompiledExpression, then we only want to use
                # our own definition, and not call ExpressionMetaClass.__new__.
                return type.__new__(mcs, class_name, bases, dict_)

        # Now we need to remove our overloaded Expression from bases,
        # and add the backend Expression type.
        bases = list(bases)
        bases.remove(Expression)
        bases.append(backend.Expression)

        # The if-test might be redundant as users should never define
        # Expression subclasses which inherit from OverloadedType.
        # In fact it might just be better to raise an error if it does.
        if OverloadedType not in bases:
            # Time to add our OverloadedType as a baseclass,
            # as well as its constructor to the
            # new class constructor.
            bases.append(OverloadedType)
            user_init = dict_.pop("__init__", None)

            def __init__(self, *args, **kwargs):
                """Overloaded init method of user-defined Expression classes.

                """
                OverloadedType.__init__(self, *args, **kwargs)
                if user_init is not None:
                    user_init(self, *args, **kwargs)

            dict_["__init__"] = __init__

        bases = tuple(bases)

        # Pass everything along to the backend metaclass.
        # This should return a new user-defined Expression 
        # subclass that inherit from OverloadedType.
        original = _backend_ExpressionMetaClass.__new__(mcs, class_name, bases, dict_)

        return original

    def __call__(cls, *args, **kwargs):
        """Handles instantiation and initialization of Expression instances.

        """
        # Calls type.__call__ as normal (backend does not override __call__).
        # Thus __new__ of cls is invoked.
        out = _backend_ExpressionMetaClass.__call__(cls, *args, **kwargs)

        # Since the class we create is not a subclass of our overloaded Expression 
        # (where __new__ was invoked), type.__call__ will not initialize
        # the newly created instance. Hence we must do so "manually".
        if not isinstance(out, cls):
            out.__init__(*args, **kwargs)
        return out


def create_compiled_expression(original, cppcode, *args, **kwargs):
    """Creates an overloaded CompiledExpression type from the supplied 
    CompiledExpression instance.

    The argument `original` will be an (uninitialized) CompiledExpression instance,
    which we extract the type from to build our own corresponding overloaded
    CompiledExpression type.

    Args:
        original (:obj:`CompiledExpression`): The original CompiledExpression instance.
        cppcode (:obj:`str`): The cppcode used to define the Expression.
        *args: Extra arguments are just passed along to the backend handlers.
        **kwargs: Keyword arguments are also just passed to backend handlers.

    Returns:
        :obj:`type`: An overloaded CompiledExpression type.

    """
    bases = (OverloadedType, type(original))

    def __init__(self, cppcode, *args, **kwargs):
        """Init method of our overloaded CompiledExpression classes.

        """
        OverloadedType.__init__(self, *args, **kwargs)
        type(original).__init__(self, cppcode, *args, **kwargs)

    return type.__new__(OverloadedExpressionMetaClass, "CompiledExpression", bases, {"__init__": __init__})


class Expression(OverloadedType, backend.Expression):
    """Overloaded Expression class.

    This class has only two functions:
        1. It overloads Expression.__new__ to create overloaded CompiledExpressions.
        
        2. It assigns an overloaded metaclass to all user-defined subclasses of Expression.

    Note: At any given point there are no classes that have this class as a base.

    """
    __metaclass__ = OverloadedExpressionMetaClass
    def __new__(cls, cppcode=None, *args, **kwargs):
        original = backend.Expression.__new__(cls, cppcode, *args, **kwargs)
        return object.__new__(create_compiled_expression(original, cppcode, *args, **kwargs))


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


# Create a default Tape so the user does not have to.
_working_tape = Tape()