import backend
import ufl

from .constant import Constant
from .function import Function
from .expression import Expression

from pyadjoint.tape import get_working_tape
from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.block import Block

# TODO: Might need/want some way of creating a new DirichletBCBlock if DirichletBC is assigned
#       new boundary values/function.


class DirichletBC(OverloadedType, backend.DirichletBC):
    def __init__(self, *args, **kwargs):
        super(DirichletBC, self).__init__(*args, **kwargs)

        # Pop kwarg to pass the kwargs check in backend.DirichletBC.__init__.
        self.annotate_tape = kwargs.pop("annotate_tape", True)
        
        backend.DirichletBC.__init__(self, *args, **kwargs)

        if self.annotate_tape:
            tape = get_working_tape()

            # Since DirichletBC behaves differently based on number of
            # args and arg types, we pass all args to block
            block = DirichletBCBlock(self, *args)
            
            tape.add_block(block)
            block.add_output(self.block_output)

    def _ad_create_checkpoint(self):
        return self

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint


class DirichletBCBlock(Block):
    def __init__(self, bc, *args):
        Block.__init__(self)
        self.bc = bc
        self.function_space = args[0]
        self.parent_space = self.function_space
        if hasattr(self.function_space, "_ad_parent_space"):
            self.parent_space = self.function_space._ad_parent_space

        if len(args) >= 2 and isinstance(args[1], OverloadedType):
            self.add_dependency(args[1].block_output)
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

    def evaluate_adj(self):
        adj_inputs = self.get_outputs()[0].get_adj_output()

        if adj_inputs is None:
            return

        for adj_input in adj_inputs:
            for block_output in self.get_dependencies():
                c = block_output.output
                if isinstance(c, Constant):
                    # Constants have float adj values.
                    block_output.add_adj_output(adj_input.sum())
                elif isinstance(c, Function):
                    # TODO: This gets a little complicated.
                    #       The function may belong to a different space,
                    #       and with `Function.set_allow_extrapolation(True)`
                    #       you can even use the Function outside its domain.
                    # For now we will just assume the FunctionSpace is the same for
                    # the BC and the Function.
                    assigner = backend.FunctionAssigner(c.function_space(), self.bc.function_space())
                    adj_output = backend.Function(c.function_space())
                    adj_value = backend.Function(self.parent_space)
                    adj_input.apply(adj_value.vector())
                    # TODO: This is not a general solution
                    assigner.assign(adj_output, adj_value.sub(int(self.bc.function_space().component()[0])))
                    block_output.add_adj_output(adj_output.vector())

    def evaluate_tlm(self):
        output = self.get_outputs()[0]

        for block_output in self.get_dependencies():
            tlm_input = block_output.tlm_value
            if tlm_input is None:
                continue

            if isinstance(block_output.output, backend.Function):
                m = backend.Function(block_output.output.function_space(), tlm_input)
            else:
                m = tlm_input

            #m = backend.project(m, self.function_space)
            m = backend.DirichletBC(self.bc.function_space(), m, *self.bc.domain_args)
            output.add_tlm_output(m)

    def evaluate_hessian(self):
        # TODO: Implement
        hessian_inputs = self.get_outputs()[0].hessian_value

        if hessian_inputs is None:
            return

        for hessian_input in hessian_inputs:
            for block_output in self.get_dependencies():
                c = block_output.output
                if isinstance(c, Constant):
                    block_output.add_hessian_output(hessian_input.sum())
                elif isinstance(c, Function):
                    # TODO: This gets a little complicated.
                    #       See evalute_adj method above.
                    assigner = backend.FunctionAssigner(c.function_space(), self.bc.function_space())
                    hessian_output = backend.Function(c.function_space())
                    hessian_value = backend.Function(self.parent_space)
                    hessian_input.apply(hessian_value.vector())
                    # TODO: This is not a general solution
                    assigner.assign(hessian_output, hessian_value.sub(int(self.bc.function_space().component()[0])))
                    block_output.add_hessian_output(hessian_output.vector())

    def recompute(self):
        # There is nothing to be recomputed.
        pass


