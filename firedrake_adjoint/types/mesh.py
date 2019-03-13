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


register_overloaded_type(MeshGeometry, backend.mesh.MeshGeometry)


class MeshInputBlock(Block):
    """
    Block which links a MeshGeometry to its coordinates, which is a firedrake
    function.
    """
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
        mesh = self.get_dependencies()[0].saved_output
        return mesh.coordinates


class MeshOutputBlock(Block):
    """
    Block which is called when the coordinates of a mesh is changed.
    """
    def __init__(self, func, mesh):
        super(MeshOutputBlock, self).__init__()
        self.add_dependency(func.block_variable)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return
        self.get_dependencies()[0].add_adj_output(adj_value)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        self.get_outputs()[0].add_tlm_output(tlm_input)
        return None

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        self.get_dependencies()[0].add_hessian_output(hessian_input)
        return None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        vector = self.get_dependencies()[0].saved_output
        mesh = vector.function_space().mesh()
        mesh.coordinates.assign(vector, annotate=False)
        self.get_outputs()[0].checkpoint = mesh._ad_create_checkpoint()
        return None


def UnitSquareMesh(*args, **kwargs):
    """ This routine wraps a UnitSquareMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.UnitSquareMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def IntervalMesh(*args, **kwargs):
    """ This routine wraps a IntervalMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.IntervalMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def UnitIntervalMesh(*args, **kwargs):
    """ This routine wraps a UnitIntervalMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.UnitIntervalMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def PeriodicIntervalMesh(*args, **kwargs):
    """ This routine wraps a PeriodicIntervalMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.PeriodicIntervalMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def PeriodicUnitIntervalMesh(*args, **kwargs):
    """ This routine wraps a PeriodicUnitIntervalMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.PeriodicUnitIntervalMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def OneElementThickMesh(*args, **kwargs):
    """ This routine wraps a OneElementThickMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.OneElementThickMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def UnitTriangleMesh(*args, **kwargs):
    """ This routine wraps a UnitTriangleMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.UnitTriangleMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def RectangleMesh(*args, **kwargs):
    """ This routine wraps a RectangleMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.RectangleMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def SquareMesh(*args, **kwargs):
    """ This routine wraps a SquareMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.SquareMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def PeriodicRectangleMesh(*args, **kwargs):
    """ This routine wraps a PeriodicRectangleMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.PeriodicRectangleMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def PeriodicUnitSquareMesh(*args, **kwargs):
    """ This routine wraps a PeriodicUnitSquareMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.PeriodicUnitSquareMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def CircleManifoldMesh(*args, **kwargs):
    """ This routine wraps a CircleManifoldMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.CircleManifoldMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def BoxMesh(*args, **kwargs):
    """ This routine wraps a BoxMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.BoxMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def CubeMesh(*args, **kwargs):
    """ This routine wraps a CubeMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.CubeMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def UnitCubeMesh(*args, **kwargs):
    """ This routine wraps a UnitCubeMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.UnitCubeMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def IcosahedralSphereMesh(*args, **kwargs):
    """ This routine wraps a IcosahedralSphereMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.IcosahedralSphereMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def UnitIcosahedralSphereMesh(*args, **kwargs):
    """ This routine wraps a UnitIcosahedralSphereMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.UnitIcosahedralSphereMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def OctahedralSphereMesh(*args, **kwargs):
    """ This routine wraps a OctahedralSphereMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.OctahedralSphereMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def UnitOctahedralSphereMesh(*args, **kwargs):
    """ This routine wraps a UnitOctahedralSphereMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.UnitOctahedralSphereMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def CubedSphereMesh(*args, **kwargs):
    """ This routine wraps a CubedSphereMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.CubedSphereMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def UnitCubedSphereMesh(*args, **kwargs):
    """ This routine wraps a UnitCubedSphereMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.UnitCubedSphereMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def TorusMesh(*args, **kwargs):
    """ This routine wraps a TorusMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.TorusMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def CylinderMesh(*args, **kwargs):
    """ This routine wraps a CylinderMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.CylinderMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output


def PartiallyPeriodicRectangleMesh(*args, **kwargs):
    """ This routine wraps a PartiallyPeriodicRectangleMesh. The purpose is to record changes
    in the computational mesh, so that one can compute the corresponding
    shape derivatives.
    """
    with stop_annotating():
        output = backend.PartiallyPeriodicRectangleMesh(*args, **kwargs)
    output = create_overloaded_object(output)

    return output
