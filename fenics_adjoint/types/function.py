import backend
from pyadjoint.tape import OverloadedType, Block, get_working_tape


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
        
        return super(Function, self).assign(other, *args, **kwargs)

    def _ad_create_checkpoint(self):
        return self.copy(deepcopy=True)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint


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

