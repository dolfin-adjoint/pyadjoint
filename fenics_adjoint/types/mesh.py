import backend
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating, no_annotations
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType, FloatingType, register_overloaded_type
from ..shapead_transformations import vector_boundary_to_mesh, vector_mesh_to_boundary


@register_overloaded_type
class Mesh(OverloadedType, backend.Mesh):
    def __init__(self, *args, **kwargs):
        # Calling constructer
        super(Mesh, self).__init__(*args, **kwargs)
        backend.Mesh.__init__(self, *args, **kwargs)
        if not len(args) == 0:
            # If the mesh is not empty, save the original coordiantes
            self.org_mesh_coords = self.coordinates().copy()
        else:
            self.org_mesh_coords = None

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


@register_overloaded_type
class BoundaryMesh(FloatingType, backend.BoundaryMesh):
    def __init__(self, *args, **kwargs):
        # Calling constructer
        super(BoundaryMesh, self).__init__(*args,
                                           block_class=BoundaryMeshBlock,
                                           _ad_args=args,
                                           _ad_floating_active=True,
                                           annotate=kwargs.pop("annotate", True),
                                           **kwargs)
        backend.BoundaryMesh.__init__(self, *args, **kwargs)

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


@register_overloaded_type
class UnitSquareMesh(OverloadedType, backend.UnitSquareMesh):
    def __init__(self, *args, **kwargs):
        # Calling constructer
        super(UnitSquareMesh, self).__init__(*args, **kwargs)
        backend.UnitSquareMesh.__init__(self, *args, **kwargs)
        self.org_mesh_coords = self.coordinates().copy()

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


@register_overloaded_type
class UnitCubeMesh(OverloadedType, backend.UnitCubeMesh):
    def __init__(self, *args, **kwargs):
        # Calling constructer
        super(UnitCubeMesh, self).__init__(*args, **kwargs)
        backend.UnitCubeMesh.__init__(self, *args, **kwargs)
        self.org_mesh_coords = self.coordinates().copy()

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


@register_overloaded_type
class UnitIntervalMesh(OverloadedType, backend.UnitIntervalMesh):
    def __init__(self, *args, **kwargs):

        # Calling constructer
        super(UnitIntervalMesh, self).__init__(*args, **kwargs)
        backend.UnitIntervalMesh.__init__(self, *args, **kwargs)
        self.org_mesh_coords = self.coordinates().copy()

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


@register_overloaded_type
class IntervalMesh(OverloadedType, backend.IntervalMesh):
    def __init__(self, *args, **kwargs):

        # Calling constructer
        super(IntervalMesh, self).__init__(*args, **kwargs)
        backend.IntervalMesh.__init__(self, *args, **kwargs)
        self.org_mesh_coords = self.coordinates().copy()

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


@register_overloaded_type
class BoxMesh(OverloadedType, backend.BoxMesh):
    def __init__(self, *args, **kwargs):

        # Calling constructer
        super(BoxMesh, self).__init__(*args, **kwargs)
        backend.BoxMesh.__init__(self, *args, **kwargs)
        self.org_mesh_coords = self.coordinates().copy()

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


@register_overloaded_type
class RectangleMesh(OverloadedType, backend.RectangleMesh):
    def __init__(self, *args, **kwargs):

        # Calling constructer
        super(RectangleMesh, self).__init__(*args, **kwargs)
        backend.RectangleMesh.__init__(self, *args, **kwargs)
        self.org_mesh_coords = self.coordinates().copy()

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


@register_overloaded_type
class SubMesh(OverloadedType, backend.SubMesh):
    def __init__(self, *args, **kwargs):
        # Calling constructer
        super(SubMesh, self).__init__(*args, **kwargs)
        backend.SubMesh.__init__(self, *args, **kwargs)
        self.org_mesh_coords = self.coordinates().copy()

    def _ad_create_checkpoint(self):
        return self.coordinates().copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates()[:] = checkpoint
        return self


__backend_ALE_move = backend.ALE.move


def move(mesh, vector, **kwargs):
    annotate = annotate_tape(kwargs)
    reset = kwargs.pop("reset_mesh", False)
    if reset:
        mesh.coordinates()[:] = mesh.org_mesh_coords
        mesh.block_variable = mesh.original_block_variable
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
        vector = self.get_dependencies()[1].saved_output

        backend.ALE.move(mesh, vector, annotate=False)

        self.get_outputs()[0].checkpoint = mesh._ad_create_checkpoint()


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

        f = backend.Function(backend.VectorFunctionSpace(self.get_outputs()[0].saved_output, "CG", 1))
        f.vector()[:] = adj_value
        adj_value = vector_boundary_to_mesh(f, self.get_dependencies()[0].saved_output)
        self.get_dependencies()[0].add_adj_output(adj_value.vector())

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return

        tlm_output = vector_mesh_to_boundary(tlm_input, self.get_outputs()[0].saved_output)
        self.get_outputs()[0].add_tlm_output(tlm_output)

    @no_annotations
    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return

        f = backend.Function(backend.VectorFunctionSpace(self.get_outputs()[0].saved_output, "CG", 1))
        f.vector()[:] = hessian_input
        hessian_value = vector_boundary_to_mesh(f, self.get_dependencies()[0].saved_output)
        self.get_dependencies()[0].add_hessian_output(hessian_value.vector())

    @no_annotations
    def recompute(self):
        mesh = self.get_dependencies()[0].saved_output

        b_mesh = BoundaryMesh(mesh, self.mesh_type, annotate=False)

        self.get_outputs()[0].checkpoint = b_mesh._ad_create_checkpoint()
