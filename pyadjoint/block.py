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

    def recompute(self):
        """This method must be overriden.

        The method should implement a routine for recomputing the block in the forward model.

        """
        raise NotImplementedError

    def create_graph(self, G, pos, scale=1.0):

        # Edges for block dependencies
        for xpos, dep in enumerate(self.get_dependencies()):
            G.add_edge(id(dep), id(self))
            G.edge[id(dep)][id(self)]['label'] = "dep"
            if "label" not in G.node[id(dep)]:
                G.node[id(dep)]['label'] = str(dep)
                G.node[id(dep)]['node_color'] = "r"
                G.node[id(dep)]['position'] = (scale*(0.1*xpos), scale*(-pos+0.5))

        # Edges for block outputs
        for xpos, out in enumerate(self.get_outputs()):
            G.add_edge(id(self), id(out))
            G.edge[id(self)][id(out)]['label'] = "out"
            if "label" not in G.node[id(out)]:
                G.node[id(out)]['label'] = str(out)
                G.node[id(out)]['node_color'] = "r"
                G.node[id(out)]['position'] = (scale*(0.1*xpos), scale*(-pos-0.5))

        # Set properties for Block node
        G.node[id(self)]['label'] = str(self)
        G.node[id(self)]['node_color'] = "b"
        G.node[id(self)]['position'] = (0, scale*(-pos))


