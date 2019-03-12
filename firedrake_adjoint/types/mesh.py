import backend
from pyadjoint.block import Block
from pyadjoint.overloaded_type import (OverloadedType, register_overloaded_type,
                                       create_overloaded_object)
from pyadjoint.tape import no_annotations, stop_annotating
from .function import Function


@register_overloaded_type
class MeshGeometry(OverloadedType, backend.mesh.MeshGeometry):
    def __init__(self, *args, **kwargs):
        super(MeshGeometry, self).__init__(*args,
                                           **kwargs)
        backend.mesh.MeshGeometry.__init__(self, *args, **kwargs)

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

    @no_annotations
    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates.assign(checkpoint)
        return self

    @backend.utils.cached_property
    def _coordinates_function(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        self.init()
        coordinates_fs = self._coordinates.function_space()
        V = backend.functionspaceimpl.WithGeometry(coordinates_fs, self)
        f = Function(V, val=self._coordinates,
                     block_class=MeshInputBlock,
                     _ad_floating_active=True,
                     _ad_args=[self],
                     _ad_output_args=[self],
                     _ad_outputs=[self],
                     output_block_class=MeshOutputBlock)
        return f

register_overloaded_type(MeshGeometry, backend.mesh.MeshGeometry)


def UnitSquareMesh(*args, **kwargs):
    """ This routine wraps a UnitSquareMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.UnitSquareMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


class MeshInputBlock(Block):
    def __init__(self, mesh):
        super(MeshInputBlock, self).__init__()
        self.add_dependency(mesh.block_variable)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return None

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return None

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self.mesh.coordinates


class MeshOutputBlock(Block):
    def __init__(self, func, mesh):
        super(MeshOutputBlock, self).__init__()
        self._ad_mesh = mesh
        self.add_dependency(func.block_variable)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return
        self.get_dependencies()[0].add_adj_output(adj_value)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return None

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self._ad_mesh
