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
        
        Will also save the output if it has not been saved before. Which should only happen if the
        BlockVariable was not created by a Block (but by the user).

        Args:
            dep (:class:`BlockVariable`): The object to be added.

        """
        if dep not in self._dependencies:  # Can be optimized if we have need for huge lists.
            dep.will_add_as_dependency()
            self._dependencies.append(dep)

    def get_dependencies(self):
        """Returns the list of dependencies.

        Returns:
            :obj:`list`: A list of :class:`BlockVariable` instances.

        """
        return self._dependencies

    def add_output(self, obj):
        """Adds object to the block output list.

        Will also save the output.

        Args:
            obj (:class:`BlockVariable`): The object to be added.

        """
        obj.will_add_as_output()
        self._outputs.append(obj)

    def get_outputs(self):
        """Returns the list of block outputs.

        Returns:
            :obj:`list`: A list of :class:`BlockVariable` instances.

        """
        return self._outputs

    def reset_variables(self, types=None):
        """Resets all adjoint variables in the block dependencies and outputs.

        """
        types = ("adjoint") if types is None else types

        for dep in self._dependencies:
            dep.reset_variables(types)

        for output in self._outputs:
            output.reset_variables(types)

    def evaluate_adj(self):
        """This method must be overriden.
        
        The method should implement a routine for evaluating the adjoint of the block.

        """
        raise NotImplementedError

    def evaluate_tlm(self):
        """This method must be overridden.
        
        The method should implement a routine for computing the tangent linear model of the block.
        Using BlockVariable.tlm_value to propagate TLM information.
        
        """
        raise NotImplementedError

    def evaluate_hessian(self):
        """This method must be overridden.

        The method should implement a routine for evaluating the hessian of the block.
        It is preferable that a "Forward-over-Reverse" scheme is used. Thus the hessians
        are evaluated in reverse (starting with the last block on the tape). Using the 
        BlockVariable.hessian_value to propagate hessian information.

        """
        raise NotImplementedError

    def recompute(self):
        """This method must be overriden.

        The method should implement a routine for recomputing the block in the forward model.
        
        Currently the designed way of doing the recomputing is to use the saved outputs/checkpoints in the
        BlockVariable dependencies, and write to the saved output/checkpoint of the BlockOuput outputs. Thus the
        recomputes are always working with the checkpoints. However I welcome suggestions of other ways to implement
        the recompute.

        """
        raise NotImplementedError
