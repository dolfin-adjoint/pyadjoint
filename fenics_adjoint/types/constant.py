import backend
from pyadjoint.tape import get_working_tape
from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.block import Block
from .types import create_overloaded_object

class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)

    def assign(self, *args, **kwargs):
        annotate_tape = kwargs.pop("annotate_tape", True)
        if annotate_tape:
            other = args[0]
            if not isinstance(other, OverloadedType):
                other = create_overloaded_object(other)

            block = AssignBlock(self, other)
            tape = get_working_tape()
            tape.add_block(block)

        ret = backend.Constant.assign(self, *args, **kwargs)

        if annotate_tape:
            block.add_output(self.create_block_output())

        return ret

    def get_derivative(self, options={}):
        return Constant(self.get_adj_output())

    def adj_update_value(self, value):
        self.original_block_output.checkpoint = value._ad_create_checkpoint()

    def _ad_create_checkpoint(self):
        if self.ufl_shape == ():
            return Constant(self)
        return Constant(self.values())

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return Constant(self*other)

    def _ad_add(self, other):
        return Constant(self+other)

    def _ad_dot(self, other):
        return sum(self.values()*other.values())


class AssignBlock(Block):
    def __init__(self, func, other):
        super(AssignBlock, self).__init__()
        self.add_dependency(func.get_block_output())
        self.add_dependency(other.get_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        self.get_dependencies()[1].add_adj_output(adj_input)

    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[1].tlm_value
        self.get_outputs()[0].add_tlm_output(tlm_input)

    def evaluate_hessian(self):
        hessian_input = self.get_outputs()[0].hessian_value
        self.get_dependencies()[1].add_hessian_output(hessian_input)

    def recompute(self):
        deps = self.get_dependencies()
        other_bo = deps[1]

        backend.Constant.assign(self.get_outputs()[0].get_saved_output(), other_bo.get_saved_output())
