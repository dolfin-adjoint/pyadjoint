import backend
import ufl
from pyadjoint.block import Block
from pyadjoint.overloaded_type import (FloatingType, register_overloaded_type,
                                       create_overloaded_object)
from pyadjoint.tape import (annotate_tape, get_working_tape, no_annotations, stop_annotating)


@register_overloaded_type
class MeshGeometry(FloatingType, backend.mesh.MeshGeometry):
    def __init__(self, *args, **kwargs):
        super(MeshGeometry, self).__init__(*args, block_class=MeshBlock,
                                           **kwargs)
        backend.mesh.MeshGeometry.__init__(self, *args, **kwargs)
        import pdb; pdb.set_trace()
        self.coordinates
        self.org_mesh_coords = self.coordinates.copy(deepcopy=True)

    @classmethod
    def _ad_init_object(cls, obj):
        callback = obj._callback
        r = cls.__new__(cls, obj.coordinates.ufl_element())
        r._topology = obj._topology
        r._callback = callback
        r.coordinates.ufl_element()
        return r
        
    def _ad_create_checkpoint(self):
        return self.coordinates.copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.obj.coordinates()[:] = checkpoint
        return self

    @backend.utils.cached_property
    def _coordinates_function(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        import backend.functionspaceimpl as functionspaceimpl
        import backend.function as function
        backend.mesh.MeshGeometry.init(self)
        coordinates_fs = self._coordinates.function_space()
        V = functionspaceimpl.WithGeometry(coordinates_fs, self)
        f = function.Function(V, val=self._coordinates)
        return create_overloaded_object(f)

register_overloaded_type(MeshGeometry, backend.mesh.MeshGeometry)


def UnitSquareMesh(*args, **kwargs):
    """ This routine wraps a UnitSquareMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding 
    shape derivatives.
    """
    annotate = annotate_tape(kwargs)

    with stop_annotating():
        output = backend.UnitSquareMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    if annotate:
        block = MeshBlock(output)
        tape = get_working_tape()
        tape.add_block(block)

        block.add_output(output.block_variable)
    return output


class MeshBlock(Block):
    def __init__(self, mesh, **kwargs):
        super(MeshBlock, self).__init__(**kwargs)
        self.add_dependency(mesh.block_variable)

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return

        self.get_dependencies()[1].add_adj_output(adj_value)

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_input = self.get_dependencies()[1].tlm_value
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
    def recompute(self, markings=False):
        mesh = self.get_dependencies()[0].saved_output
        coordinates = self.get_dependencies()[1].saved_output

        mesh.coordinates = coordinates

        self.get_outputs()[0].checkpoint = mesh._ad_create_checkpoint()
