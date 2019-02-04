from pyadjoint.tape import (get_working_tape, annotate_tape,
                            no_annotations)
from pyadjoint.block import Block
from .types import BoundaryMeshType


def BoundaryMesh(*args, **kwargs):
    annotate = annotate_tape(kwargs)
    output = BoundaryMeshType(*args, **kwargs)
    if annotate:
        block = BoundaryMeshBlock(*args, **kwargs)
        tape = get_working_tape()
        tape.add_block(block)
        block.add_output(output.block_variable)
    return output


class BoundaryMeshBlock(Block):
    def __init__(self, mesh, mesh_type, **kwargs):
        super(BoundaryMeshBlock, self).__init__(**kwargs)
        self.add_dependency(mesh.block_variable)
        self.mesh_type = mesh_type

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return

        self.get_dependencies()[0].add_adj_output(adj_value)

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        self.get_outputs()[0].add_tlm_output(tlm_input)

    @no_annotations
    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return

        self.get_dependencies()[1].add_hessian_output(hessian_input)

    @no_annotations
    def recompute(self):
        mesh = self.get_dependencies()[0].saved_output

        b_mesh = BoundaryMesh(mesh, self.mesh_type, annotate=False)

        self.get_outputs()[0].checkpoint = b_mesh._ad_create_checkpoint()
