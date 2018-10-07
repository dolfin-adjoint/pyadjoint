from .tape import no_annotations


class \
        Block(object):
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

    def add_dependency(self, dep, no_duplicates=False):
        """Adds object to the block dependencies.
        
        Will also save the output if it has not been saved before. Which should only happen if the
        BlockVariable was not created by a Block (but by the user).

        Args:
            dep (BlockVariable): The object to be added.
            no_duplicates (bool, optional): If True, the dependency is only added if it is not already in the list.
                Default is False.

        """
        if not no_duplicates or dep not in self._dependencies:
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

    @no_annotations
    def evaluate_adj(self, markings=False):
        outputs = self.get_outputs()
        adj_inputs = []
        has_input = False
        for output in outputs:
            adj_inputs.append(output.adj_value)
            if output.adj_value is not None:
                has_input = True

        if not has_input:
            return

        deps = self.get_dependencies()
        inputs = [bv.saved_output for bv in deps]
        relevant_dependencies = [(i, bv) for i, bv in enumerate(deps) if bv.marked_in_path or not markings]

        if len(relevant_dependencies) <= 0:
            return

        prepared = self.prepare_evaluate_adj(inputs, adj_inputs, relevant_dependencies)

        for idx, dep in relevant_dependencies:
            adj_output = self.evaluate_adj_component(inputs,
                                                     adj_inputs,
                                                     dep,
                                                     idx,
                                                     prepared)
            if adj_output is not None:
                dep.add_adj_output(adj_output)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        """Runs preparations before `evalute_adj_component` is ran.

        The return value is supplied to each of the subsequent `evaluate_adj_component` calls.
        This method is intended to be overridden for blocks that require such preparations, by default there is none.

        Args:
            inputs: The values of the inputs
            adj_inputs: The adjoint inputs
            relevant_dependencies: A list of the relevant block variables for `evaluate_adj_component`.

        Returns:
            Anything. The returned value is supplied to `evaluate_adj_component`

        """
        return None

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        """This method must be overriden.
        
        The method should implement a routine for evaluating the adjoint of the block.

        """
        raise NotImplementedError(type(self))

    def evaluate_tlm(self, markings=False):
        deps = self.get_dependencies()
        tlm_inputs = []
        has_input = False
        for dep in deps:
            tlm_inputs.append(dep.tlm_value)
            if dep.tlm_value is not None:
                has_input = True

        if not has_input:
            return

        outputs = self.get_outputs()
        inputs = [bv.saved_output for bv in deps]
        relevant_outputs = [(i, bv) for i, bv in enumerate(outputs) if bv.marked_in_path or not markings]

        if len(relevant_outputs) <= 0:
            return

        prepared = self.prepare_evaluate_tlm(inputs, tlm_inputs, relevant_outputs)

        for idx, out in relevant_outputs:
            tlm_output = self.evaluate_tlm_component(inputs,
                                                     tlm_inputs,
                                                     out,
                                                     idx,
                                                     prepared)
            if tlm_output is not None:
                out.add_tlm_output(tlm_output)

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        """Runs preparations before `evalute_tlm_component` is ran.

        The return value is supplied to each of the subsequent `evaluate_tlm_component` calls.
        This method is intended to be overridden for blocks that require such preparations, by default there is none.

        Args:
            inputs: The values of the inputs
            tlm_inputs: The tlm inputs
            relevant_outputs: A list of the relevant block variables for `evaluate_tlm_component`.

        Returns:
            Anything. The returned value is supplied to `evaluate_tlm_component`

        """
        return None

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        """This method must be overridden.
        
        The method should implement a routine for computing the tangent linear model of the block.
        
        """
        raise NotImplementedError("evaluate_tlm_component is not implemented for Block-type: {}".format(type(self)))

    @no_annotations
    def evaluate_hessian(self, markings=False):
        outputs = self.get_outputs()
        hessian_inputs = []
        adj_inputs = []
        has_input = False
        for output in outputs:
            hessian_inputs.append(output.hessian_value)
            adj_inputs.append(output.adj_value)
            if output.hessian_value is not None:
                has_input = True

        if not has_input:
            return

        deps = self.get_dependencies()
        inputs = [bv.saved_output for bv in deps]
        relevant_dependencies = [(i, bv) for i, bv in enumerate(deps) if bv.marked_in_path or not markings]

        if len(relevant_dependencies) <= 0:
            return

        prepared = self.prepare_evaluate_hessian(inputs, hessian_inputs, adj_inputs, relevant_dependencies)

        for idx, dep in relevant_dependencies:
            hessian_output = self.evaluate_hessian_component(inputs,
                                                             hessian_inputs,
                                                             adj_inputs,
                                                             dep,
                                                             idx,
                                                             relevant_dependencies,
                                                             prepared)
            if hessian_output is not None:
                dep.add_hessian_output(hessian_output)

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        """Runs preparations before `evalute_hessian_component` is ran for each relevant dependency.

        The return value is supplied to each of the subsequent `evaluate_hessian_component` calls.
        This method is intended to be overridden for blocks that require such preparations, by default there is none.

        Args:
            inputs: The values of the inputs
            hessian_inputs: The hessian inputs
            adj_inputs: The adjoint inputs
            relevant_dependencies: A list of the relevant block variables for `evaluate_hessian_component`.

        Returns:
            Anything. The returned value is supplied to `evaluate_hessian_component`

        """
        return None

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        """This method must be overridden.

        The method should implement a routine for evaluating the hessian of the block.
        It is preferable that a "Forward-over-Reverse" scheme is used. Thus the hessians
        are evaluated in reverse (starting with the last block on the tape).

        """
        raise NotImplementedError(type(self))

    def recompute(self, markings=False):
        outputs = self.get_outputs()
        for out in outputs:
            if out.is_control:
                # We assume that a Block with multiple outputs where one of them is a control
                # can not depend on any other controls.
                return

        inputs = [bv.saved_output for bv in self.get_dependencies()]
        relevant_outputs = [(i, bv) for i, bv in enumerate(outputs) if bv.marked_in_path or not markings]

        if len(relevant_outputs) <= 0:
            return

        prepared = self.prepare_recompute_component(inputs, relevant_outputs)

        for idx, out in relevant_outputs:
            output = self.recompute_component(inputs,
                                              out,
                                              idx,
                                              prepared)
            if output is not None:
                out.checkpoint = output

    def prepare_recompute_component(self, inputs, relevant_outputs):
        """Runs preparations before `recompute_component` is ran.

        The return value is supplied to each of the subsequent `recompute_component` calls.
        This method is intended to be overridden for blocks that require such preparations, by default there is none.

        Args:
            inputs: The values of the inputs
            relevant_outputs: A list of the relevant block variables for `recompute_component`.

        Returns:
            Anything. The returned value is supplied to `recompute_component`

        """
        return None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        """This method must be overriden.

        The method should implement a routine for recomputing the block in the forward model.

        Currently the designed way of doing the recomputing is to use the saved outputs/checkpoints in the
        BlockVariable dependencies, and write to the saved output/checkpoint of the BlockOuput outputs. Thus the
        recomputes are always working with the checkpoints. However I welcome suggestions of other ways to implement
        the recompute.

        """
        raise NotImplementedError
