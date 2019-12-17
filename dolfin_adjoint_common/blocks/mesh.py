from pyadjoint import Block

class ALEMoveBlock(Block):
    def __init__(self, mesh, vector, **kwargs):
        super(ALEMoveBlock, self).__init__(**kwargs)
        self.add_dependency(mesh)
        self.add_dependency(vector)

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return
        self.get_dependencies()[0].add_adj_output(adj_value.copy())
        self.get_dependencies()[1].add_adj_output(adj_value)

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_output = None

        for i in range(2):
            tlm_input = self.get_dependencies()[i].tlm_value
            if tlm_input is None:
                continue
            if tlm_output is None:
                tlm_output = tlm_input.copy(deepcopy=True)
            else:
                tlm_output.vector()[:] += tlm_input.vector()

        if tlm_output is not None:
            self.get_outputs()[0].add_tlm_output(tlm_output)

    @no_annotations
    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        self.get_dependencies()[0].add_hessian_output(hessian_input.copy())
        self.get_dependencies()[1].add_hessian_output(hessian_input)

    @no_annotations
    def recompute(self, markings=False):
        mesh = self.get_dependencies()[0].saved_output
        vector = self.get_dependencies()[1].saved_output

        self.compat.ALE.move(mesh, vector, annotate=False)

        self.get_outputs()[0].checkpoint = mesh._ad_create_checkpoint()


class BoundaryMeshBlock(Block):
    def __init__(self, mesh, mesh_type, **kwargs):
        super(BoundaryMeshBlock, self).__init__(**kwargs)
        self.add_dependency(mesh)
        self.mesh_type = mesh_type

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_value = self.get_outputs()[0].adj_value
        if adj_value is None:
            return

        f = self.compat.Function(self.compat.VectorFunctionSpace(self.get_outputs()[0].saved_output, "CG", 1))
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

        f = self.compat.Function(self.compat.VectorFunctionSpace(self.get_outputs()[0].saved_output, "CG", 1))
        f.vector()[:] = hessian_input
        hessian_value = vector_boundary_to_mesh(f, self.get_dependencies()[0].saved_output)
        self.get_dependencies()[0].add_hessian_output(hessian_value.vector())

    @no_annotations
    def recompute(self):
        mesh = self.get_dependencies()[0].saved_output

        b_mesh = BoundaryMesh(mesh, self.mesh_type, annotate=False)

        self.get_outputs()[0].checkpoint = b_mesh._ad_create_checkpoint()
