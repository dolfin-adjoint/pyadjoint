import backend
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating, no_annotations
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType

class Mesh(OverloadedType, backend.Mesh):
    def __init__(self, *args, **kwargs):
        # Calling constructer
        super(Mesh, self).__init__(*args, **kwargs)
        backend.Mesh.__init__(self, *args, **kwargs)

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


__backend_ALE_move = backend.ALE.move
def move(mesh, vector, **kwargs):
    annotate = annotate_tape(kwargs)

    if annotate:
        assert isinstance(mesh, OverloadedType)
        assert isinstance(vector, OverloadedType)
        tape = get_working_tape()
        block = ALEMoveBlock(mesh, vector, **kwargs)
        tape.add_block(block)

    with stop_annotating():
        output = __backend_ALE_move(mesh, vector)

    if annotate:
        block.add_output(mesh.create_block_variable())
    return output


backend.ALE.move = move


class ALEMoveBlock(Block):
    def __init__(self, mesh, vector, **kwargs):
        super(ALEMoveBlock, self).__init__(**kwargs)
        self.add_dependency(mesh.block_variable)
        self.add_dependency(vector.block_variable)

    @no_annotations
    def evaluate_adj(self):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return

        self.get_dependencies()[1].add_adj_output(adj_value)
        
    @no_annotations
    def recompute(self):
        mesh = self.get_dependencies()[0].saved_output
        vector = self.get_dependencies()[1].saved_output

        backend.ALE.move(mesh, vector, annotate=False)

        self.get_outputs()[0].checkpoint = mesh._ad_create_checkpoint()
