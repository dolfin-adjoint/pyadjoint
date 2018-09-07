import backend
import ufl

from . import compat
from .constant import Constant
from .function import Function
from .function_space import extract_subfunction

from pyadjoint.tape import get_working_tape, annotate_tape, no_annotations, stop_annotating
from pyadjoint.overloaded_type import OverloadedType, FloatingType
from pyadjoint.block import Block

import numpy

# TODO: Might need/want some way of creating a new DirichletBCBlock if DirichletBC is assigned
#       new boundary values/function.


class DirichletBC(FloatingType, backend.DirichletBC):
    def __init__(self, *args, **kwargs):
        super(DirichletBC, self).__init__(*args, **kwargs)

        FloatingType.__init__(self,
                              *args,
                              block_class=DirichletBCBlock,
                              _ad_args=args,
                              _ad_floating_active=True,
                              annotate=kwargs.pop("annotate", True),
                              **kwargs)

        # Call backend constructor after popped AD specific keyword args.
        backend.DirichletBC.__init__(self, *args, **kwargs)

        self._g = args[1]
        self._ad_args = args
        self._ad_kwargs = kwargs

    @no_annotations
    def apply(self, *args, **kwargs):
        for arg in args:
            if not hasattr(arg, "bcs"):
                arg.bcs = []
            arg.bcs.append(self)
        return backend.DirichletBC.apply(self, *args, **kwargs)

    def _ad_create_checkpoint(self):
        deps = self.block.get_dependencies()
        if len(deps) <= 0:
            # We don't have any dependencies so the supplied value was not an OverloadedType.
            # Most probably it was just a float that is immutable so will never change.
            return None

        return deps[0]

    def _ad_restore_at_checkpoint(self, checkpoint):
        if checkpoint is not None:
            self.set_value(checkpoint.saved_output)
        return self


def _extract_subindices(V):
    assert V.num_sub_spaces() > 0
    r = []
    for i in range(V.num_sub_spaces()):
        l = [i]
        _build_subindices(l, r, V.sub(i))
        l.pop()
    return r


def _build_subindices(l, r, V):
    if V.num_sub_spaces() <= 0:
        r.append(tuple(l))
    else:
        for i in range(V.num_sub_spaces()):
            l.append(i)
            _build_subindices(l, r, V)
            l.pop()


class DirichletBCBlock(Block):
    def __init__(self, *args):
        Block.__init__(self)
        self.function_space = args[0]
        self.parent_space = self.function_space
        if hasattr(self.function_space, "_ad_parent_space") and self.function_space._ad_parent_space is not None:
            self.parent_space = self.function_space._ad_parent_space
        self.collapsed_space = None
        if self.function_space != self.parent_space:
            self.collapsed_space = self.function_space.collapse()

        if len(args) >= 2 and isinstance(args[1], OverloadedType):
            self.add_dependency(args[1].block_variable)
        else:
            # TODO: Implement the other cases.
            #       Probably just a BC without dependencies?
            #       In which case we might not even need this Block?
            # Update: What if someone runs: `DirichletBC(V, g*g, "on_boundary")`.
            #         In this case the backend will project the product onto V.
            #         But we will have to annotate the product somehow.
            #         One solution would be to do a check and add a ProjectBlock before the DirichletBCBlock.
            #         (Either by actually running our project or by "manually" inserting a project block).
            pass

    @no_annotations
    def evaluate_adj(self):
        bc = self.get_outputs()[0].saved_output
        adj_inputs = self.get_outputs()[0].adj_value

        if adj_inputs is None:
            return

        for adj_input in adj_inputs:
            for block_variable in self.get_dependencies():
                c = block_variable.output
                if isinstance(c, Constant):
                    adj_value = backend.Function(self.parent_space)
                    adj_input.apply(adj_value.vector())
                    if self.collapsed_space is not None:
                        vec = compat.extract_bc_subvector(adj_value, self.collapsed_space, bc)
                        adj_value = compat.function_from_vector(self.collapsed_space, vec)

                    if adj_value.ufl_shape == () or adj_value.ufl_shape[0] <= 1:
                        block_variable.add_adj_output(adj_value.vector().sum())
                    else:
                        adj_output = []
                        subindices = _extract_subindices(self.function_space)
                        for indices in subindices:
                            current_subfunc = adj_value
                            prev_idx = None
                            for i in indices:
                                if prev_idx is not None:
                                    current_subfunc = current_subfunc.sub(prev_idx)
                                prev_idx = i
                            adj_output.append(current_subfunc.sub(prev_idx, deepcopy=True).vector().sum())
                        adj_vector = backend.cpp.la.Vector(backend.MPI.comm_world, len(adj_output))
                        adj_vector[:] = adj_output
                        block_variable.add_adj_output(adj_vector)
                elif isinstance(c, Function):
                    # TODO: This gets a little complicated.
                    #       The function may belong to a different space,
                    #       and with `Function.set_allow_extrapolation(True)`
                    #       you can even use the Function outside its domain.
                    # For now we will just assume the FunctionSpace is the same for
                    # the BC and the Function.
                    adj_value = backend.Function(self.parent_space)
                    adj_input.apply(adj_value.vector())
                    adj_output = compat.extract_bc_subvector(adj_value, c.function_space(), bc)
                    block_variable.add_adj_output(adj_output)

    @no_annotations
    def evaluate_tlm(self):
        output = self.get_outputs()[0]
        bc = output.saved_output

        for block_variable in self.get_dependencies():
            tlm_input = block_variable.tlm_value
            if tlm_input is None:
                continue

            # TODO: This is gonna crash for dirichletbcs with multiple dependencies (can't add two bcs)
            #       However, if there is multiple dependencies, we need to AD the expression (i.e if value=f*g then
            #       dvalue = tlm_f * g + f * tlm_g). Right now we can only handle value=f => dvalue = tlm_f.
            m = compat.create_bc(bc, value=tlm_input)
            output.add_tlm_output(m)

    @no_annotations
    def evaluate_hessian(self):
        # TODO: This is the exact same as evaluate_adj for now. Consider refactoring for no duplicate code.
        bc = self.get_outputs()[0].saved_output
        hessian_inputs = self.get_outputs()[0].hessian_value

        if hessian_inputs is None:
            return

        for hessian_input in hessian_inputs:
            for block_variable in self.get_dependencies():
                c = block_variable.output
                if isinstance(c, Constant):
                    hessian_value = backend.Function(self.parent_space)
                    hessian_input.apply(hessian_value.vector())
                    if self.collapsed_space is not None:
                        vec = compat.extract_bc_subvector(hessian_value, self.collapsed_space, bc)
                        hessian_value = compat.function_from_vector(self.collapsed_space, vec)

                    if hessian_value.ufl_shape == () or hessian_value.ufl_shape[0] <= 1:
                        block_variable.add_hessian_output(hessian_value.vector().sum())
                    else:
                        hessian_output = []
                        subindices = _extract_subindices(self.function_space)
                        for indices in subindices:
                            current_subfunc = hessian_value
                            prev_idx = None
                            for i in indices:
                                if prev_idx is not None:
                                    current_subfunc = current_subfunc.sub(prev_idx)
                                prev_idx = i
                            hessian_output.append(current_subfunc.sub(prev_idx, deepcopy=True).vector().sum())
                        hessian_vector = backend.cpp.la.Vector(backend.MPI.comm_world, len(hessian_output))
                        hessian_vector[:] = hessian_output
                        block_variable.add_hessian_output(hessian_vector)
                elif isinstance(c, Function):
                    # TODO: This gets a little complicated.
                    #       The function may belong to a different space,
                    #       and with `Function.set_allow_extrapolation(True)`
                    #       you can even use the Function outside its domain.
                    # For now we will just assume the FunctionSpace is the same for
                    # the BC and the Function.
                    hessian_value = backend.Function(self.parent_space)
                    hessian_input.apply(hessian_value.vector())
                    hessian_output = compat.extract_bc_subvector(hessian_value, c.function_space(), bc)
                    block_variable.add_hessian_output(hessian_output)

    @no_annotations
    def recompute(self):
        # There is nothing to do. The checkpoint is weak,
        # so it changes automatically with the dependency checkpoint.
        return

    def __str__(self):
        return "DirichletBC block"
