import backend
import sys
from pyadjoint.block import Block
from pyadjoint.overloaded_type import (OverloadedType, register_overloaded_type,
                                       create_overloaded_object)
from pyadjoint.tape import no_annotations, stop_annotating
from .function import Function


__all__ = ["Mesh"] + backend.utility_meshes.__all__


@register_overloaded_type
class MeshGeometry(OverloadedType, backend.mesh.MeshGeometry):
    def __init__(self, *args, **kwargs):
        super(MeshGeometry, self).__init__(*args,
                                           **kwargs)
        backend.mesh.MeshGeometry.__init__(self, *args, **kwargs)

    def _ad_overloaded_init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _ad_init_object(cls, obj):
        obj.__class__ = cls
        obj._ad_overloaded_init()
        if "_coordinates_function" in obj.__dict__:
            coordinates = obj.coordinates
            del obj.__dict__["_coordinates_function"]
            obj.coordinates.interpolate(coordinates)
        return obj

    def _ad_create_checkpoint(self):
        return self.coordinates.copy(deepcopy=True)

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


class MeshInputBlock(Block):
    """
    Block which links a MeshGeometry to its coordinates, which is a firedrake
    function.
    """
    def __init__(self, mesh):
        super(MeshInputBlock, self).__init__()
        self.add_dependency(mesh)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        mesh = self.get_dependencies()[0].saved_output
        return mesh.coordinates


class MeshOutputBlock(Block):
    """
    Block which is called when the coordinates of a mesh is changed.
    """
    def __init__(self, func, mesh):
        super(MeshOutputBlock, self).__init__()
        self.add_dependency(func)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        vector = self.get_dependencies()[0].saved_output
        mesh = vector.function_space().mesh()
        mesh.coordinates.assign(vector, annotate=False)
        return mesh._ad_create_checkpoint()


def overloaded_mesh(constructor):
    def inner(*args, **kwargs):
        "Creating an overloaded Mesh that will be used in shape optimization"
        with stop_annotating():
            mesh = constructor(*args, **kwargs)
        return create_overloaded_object(mesh)
    return inner


thismod = sys.modules[__name__]
for name in backend.utility_meshes.__all__:
    setattr(thismod, name, overloaded_mesh(getattr(backend, name)))
    mod = getattr(thismod, name)
    mod.__doc__ = getattr(backend, name).__doc__

Mesh = overloaded_mesh(backend.Mesh)
Mesh.__doc__ = backend.Mesh.__doc__
