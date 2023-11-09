from .tape import no_annotations
from html import escape


class Block(object):
    """Base class for all Tape Block types.

    Each instance of a Block type represents an elementary operation in the
    forward model.

    Abstract methods
        :func:`evaluate_adj`

    """
    __slots__ = ['_dependencies', '_outputs', 'block_helper']
    pop_kwargs_keys = []

    def __init__(self, ad_block_tag=None):
        self._dependencies = []
        self._outputs = []
        self.block_helper = None
        self.tag = ad_block_tag

    @classmethod
    def pop_kwargs(cls, kwargs):
        """Takes in a dictionary of keyword arguments,
        and pops the ones used by the Block-subclass `cls`
        """
        keys = cls.pop_kwargs_keys
        d = {}
        for k in keys:
            if k in kwargs:
                d[k] = kwargs.pop(k)
        return d

    def reset(self):
        if self.block_helper is not None:
            self.block_helper.reset()

    def add_dependency(self, dep, no_duplicates=False):
        """Adds object to the block dependencies.

        Will also save the output if it has not been saved before. Which should only happen if the
        BlockVariable was not created by a Block (but by the user).

        Args:
            dep (OverloadedType): The object to be added.
            no_duplicates (bool, optional): If True, the dependency is only added if it is not already in the list.
                Default is False.

        """
        if not no_duplicates or dep.block_variable not in self._dependencies:
            dep._ad_will_add_as_dependency()
            self._dependencies.append(dep.block_variable)

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
        """Computes the adjoint action and stores the result in the `adj_value` attribute of the dependencies.

        This method will by default call the `evaluate_adj_component` method for each dependency.

        Args:
            markings (bool): If True, then each block_variable will have set `marked_in_path` attribute indicating
                whether their adjoint components are relevant for computing the final target adjoint values.
                Default is False.

        """
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
        """This method should be overridden.

        The method should implement a routine for evaluating the adjoint of the block that corresponds to
        one dependency.
        If one considers the adjoint action a vector right multiplied with the Jacobian matrix,
        then this method should return one entry in the resulting product, where the entry
        returned is decided by the argument `idx`.

        Args:
            inputs (list): A list of the saved input values, determined by the dependencies list.
            adj_inputs (list): A list of the adjoint input values, determined by the outputs list.
            block_variable (BlockVariable): The block variable of the dependency corresponding to index `idx`.
            idx (int): The index of the component to compute.
            prepared (object): Anything returned by the prepare_evaluate_adj method. Default is None.

        Returns:
            An object of a type consistent with the adj_value type of `block_variable`: The resulting product.

        """
        raise NotImplementedError(type(self))

    @no_annotations
    def evaluate_tlm(self, markings=False):
        """Computes the tangent linear action and stores the result in the `tlm_value` attribute of the outputs.

        This method will by default call the `evaluate_tlm_component` method for each output.

        Args:
            markings (bool): If True, then each block_variable will have set `marked_in_path` attribute indicating
                whether their tlm components are relevant for computing the final target tlm values.
                Default is False.

        """
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
        """This method should be overridden.

        The method should implement a routine for computing the tangent linear model of the block that corresponds to
        one output.
        If one considers the tangent linear action as a Jacobian matrix multiplied with a vector,
        then this method should return one entry in the resulting product, where the entry returned
        is decided by the argument `idx`.

        Args:
            inputs (list): A list of the saved input values, determined by the dependencies list.
            tlm_inputs (list): A list of the tlm input values, determined by the dependencies list.
            block_variable (BlockVariable): The block variable of the output corresponding to index `idx`.
            idx (int): The index of the component to compute.
            prepared (object): Anything returned by the prepare_evaluate_tlm method. Default is None.

        Returns:
            An object of the same type as `block_variable.saved_output`: The resulting product.

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
        """Recomputes the overloaded function with new inputs
            and stores the results in the `checkpoint` attribute of the outputs.

        This method will by default call the `recompute_component` method for each output.

        Args:
            markings (bool): If True, then each block_variable will have set `marked_in_path` attribute indicating
                whether their checkpoints need to be recomputed for recomputing the final target function value.
                Default is False.

        """
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
        """This method must be overridden.

        The method should implement a routine for recomputing one output of the block in the forward computations.
        The output to recompute is determined by the `idx` argument, which corresponds to the index
        of the output in the outputs list.
        If the block only has a single output, then `idx` will always be 0.

        Args:
            inputs (list): A list of the saved input values, determined by the dependencies list.
            block_variable (BlockVariable): The block variable of the output corresponding to index `idx`.
            idx (int): The index of the output to compute.
            prepared (object): Anything returned by the prepare_recompute_component method. Default is None.

        Returns:
            An object of the same type as `block_variable.checkpoint` which is determined by
            `OverloadedType._ad_create_checkpoint` (often the same as `block_variable.saved_output`): The new output.

        """
        raise NotImplementedError

    def create_graph(self, G, pos):
        # Edges for block dependencies
        for xpos, dep in enumerate(self.get_dependencies()):
            G.add_edge(id(dep), id(self))
            if "label" not in G.nodes[id(dep)]:
                G.nodes[id(dep)]['label'] = escape(str(dep))
                G.nodes[id(dep)]['node_color'] = "r"
                G.nodes[id(dep)]['position'] = (0.1 * xpos, -pos + 0.5)

        # Edges for block outputs
        for xpos, out in enumerate(self.get_outputs()):
            G.add_edge(id(self), id(out))
            if "label" not in G.nodes[id(out)]:
                G.nodes[id(out)]['label'] = escape(str(out))
                G.nodes[id(out)]['node_color'] = "r"
                G.nodes[id(out)]['position'] = (0.1 * xpos, -pos - 0.5)

        # Set properties for Block node
        G.nodes[id(self)]['label'] = escape(str(self))
        G.nodes[id(self)]['node_color'] = "b"
        G.nodes[id(self)]['position'] = (0, -pos)
        G.nodes[id(self)]['shape'] = "box"
        G.nodes[id(self)]['style'] = "filled"
        G.nodes[id(self)]['fillcolor'] = "lightgrey"
