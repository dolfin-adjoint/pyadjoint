import backend
from pyadjoint.tape import get_working_tape
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType


class Function(OverloadedType, backend.Function):
    def __init__(self, *args, **kwargs):
        super(Function, self).__init__(*args, **kwargs)
        backend.Function.__init__(self, *args, **kwargs)

    def copy(self, *args, **kwargs):
        # Overload the copy method so we actually return overloaded types.
        # Otherwise we might end up getting unexpected errors later.
        c = backend.Function.copy(self, *args, **kwargs)
        return Function(c.function_space(), c.vector())

    def assign(self, other, *args, **kwargs):
        annotate_tape = kwargs.pop("annotate_tape", True)
        if annotate_tape:
            block = AssignBlock(self, other)
            tape = get_working_tape()
            tape.add_block(block)
        
        return super(Function, self).assign(other, *args, **kwargs)

    def get_derivative(self):
        adj_value = self.get_adj_output()
        return Function(self.function_space(), adj_value)

    def adj_update_value(self, value):
        if isinstance(value, backend.Function):
            super(Function, self).assign(value)
            # TODO: Consider how recomputations are done.
            #       i.e. if they use saved output or not.
            self.original_block_output.save_output()
        else:
            # TODO: Do we want to remove this? Might be useful,
            #       but the design of pyadjoint does not require
            #       such an implementation.
            
            # Assuming vector
            self.vector()[:] = value

    def _ad_mult(self, other):
        r = Function(self.function_space())
        backend.Function.assign(r, self*other)
        return r

    def _ad_add(self, other):
        r = Function(self.function_space())
        backend.Function.assign(r, self+other)
        return r

    def _ad_dot(self, other):
        return self.vector().inner(other.vector())



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

    def recompute(self):
        deps = self.get_dependencies()
        func_bo = deps[0]
        other_bo = deps[1]

        # Currently only saves other as it is usually written to
        # in solve during recomputation. However we can't save func_bo
        # as it currently holds the values it has at the end of a forward computation
        # (since the recompute deals with saved outputs and not the real outputs)
        # All this will be made much simpler when we create a new system for saving output
        # TODO: We want to always save output, not only on assigns or similar operations.
        other_bo.save_output()

        backend.Function.assign(func_bo.output, other_bo.output)
        
        self.get_outputs()[0].save_output()