import backend
import numpy
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.block import Block
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations


def SyncSum(vec):
    """ Returns sum of vec over all mpi processes.

    Each vec vector must have the same dimension for each MPI process """

    comm = backend.MPI.comm_world
    NormalsAllProcs = numpy.zeros(comm.Get_size() * len(vec), dtype=vec.dtype)
    comm.Allgather(vec, NormalsAllProcs)

    out = numpy.zeros(len(vec))
    for j in range(comm.Get_size()):
        out += NormalsAllProcs[len(vec) * j:len(vec) * (j + 1)]
    return out


def mesh_to_boundary(v, b_mesh):
    """
    Returns a the boundary representation of the CG-1 function v
    """
    # Extract the underlying volume and boundary meshes
    mesh = v.function_space().mesh()

    # We use a Dof->Vertex mapping to create a global
    # array with all DOF values ordered by mesh vertices
    DofToVert = backend.dof_to_vertex_map(v.function_space())
    VGlobal = numpy.zeros(v.vector().size())

    vec = v.vector().get_local()
    for i in range(len(vec)):
        Vert = backend.MeshEntity(mesh, 0, DofToVert[i])
        globalIndex = Vert.global_index()
        VGlobal[globalIndex] = vec[i]
    VGlobal = SyncSum(VGlobal)

    # Use the inverse mapping to se the DOF values of a boundary
    # function
    surface_space = backend.FunctionSpace(b_mesh, "CG", 1)
    surface_function = backend.Function(surface_space)
    mapa = b_mesh.entity_map(0)
    DofToVert = backend.dof_to_vertex_map(backend.FunctionSpace(b_mesh, "CG", 1))

    LocValues = surface_function.vector().get_local()
    for i in range(len(LocValues)):
        VolVert = backend.MeshEntity(mesh, 0, mapa[int(DofToVert[i])])
        GlobalIndex = VolVert.global_index()
        LocValues[i] = VGlobal[GlobalIndex]

    surface_function.vector().set_local(LocValues)
    surface_function.vector().apply('')
    return surface_function


def boundary_to_mesh(f, mesh):
    """ Take a CG1 function f defined on a surface mesh and return a
    volume vector with same values on boundary but zero in volume
    """
    b_mesh = f.function_space().mesh()
    SpaceV = backend.FunctionSpace(mesh, "CG", 1)
    SpaceB = backend.FunctionSpace(b_mesh, "CG", 1)

    F = backend.Function(SpaceV)
    GValues = numpy.zeros(F.vector().size())

    map = b_mesh.entity_map(0)  # Vertex map from boundary mesh to parent mesh
    d2v = backend.dof_to_vertex_map(SpaceB)
    v2d = backend.vertex_to_dof_map(SpaceV)

    dof = SpaceV.dofmap()
    imin, imax = dof.ownership_range()

    for i in range(f.vector().local_size()):
        GVertID = backend.Vertex(b_mesh, d2v[i]).index()  # Local Vertex ID for given dof on boundary mesh
        PVertID = map[GVertID]  # Local Vertex ID of parent mesh
        PDof = v2d[PVertID]  # Dof on parent mesh
        value = f.vector()[i]  # Value on local processor
        GValues[dof.local_to_global_index(PDof)] = value
    GValues = SyncSum(GValues)

    F.vector().set_local(GValues[imin:imax])
    F.vector().apply("")
    return F


def vector_boundary_to_mesh(boundary_func, mesh):
    """
    Transfer a Vector-CG1 function from a Boundary mesh to
    a CG-1 function living on the Parent mesh (where all interior
    values are 0). This function is only meant to be called internally in
    pyadjoint, or for verification purposes.
    """
    V = backend.VectorFunctionSpace(mesh, "CG", 1)
    vb_split = boundary_func.split(deepcopy=True)
    v_vol = []
    for vb in vb_split:
        v_vol.append(boundary_to_mesh(vb, mesh))
    scalar_to_vec = backend.FunctionAssigner(V, [v.function_space()
                                                 for v in v_vol])
    v_out = backend.Function(V)
    scalar_to_vec.assign(v_out, v_vol)
    return v_out


def vector_mesh_to_boundary(func, b_mesh):
    v_split = func.split(deepcopy=True)
    v_b = []
    for v in v_split:
        v_b.append(mesh_to_boundary(v, b_mesh))
    Vb = backend.VectorFunctionSpace(b_mesh, "CG", 1)
    vb_out = backend.Function(Vb)
    scalar_to_vec = backend.FunctionAssigner(Vb, [v.function_space() for
                                                  v in v_b])
    scalar_to_vec.assign(vb_out, v_b)
    return vb_out


def transfer_from_boundary(*args, **kwargs):
    """
    Transfers values from a CG1 function on the BoundaryMesh to its
    original mesh
    """
    annotate = annotate_tape(kwargs)
    with stop_annotating():
        output = vector_boundary_to_mesh(*args)
    output = create_overloaded_object(output)

    if annotate:
        block = SurfaceTransferBlock(args[0], args[1])
        tape = get_working_tape()
        tape.add_block(block)
        block.add_output(output.block_variable)

    return output


class SurfaceTransferBlock(Block):
    def __init__(self, func, other):
        super(SurfaceTransferBlock, self).__init__()
        self.add_dependency(func)
        self.add_dependency(func.function_space().mesh())

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        b_mesh = self.get_dependencies()[1].output
        adj_value = backend.Function(W)
        adj_value.vector()[:] = adj_input
        adj_output = vector_mesh_to_boundary(adj_value, b_mesh)
        self.get_dependencies()[0].add_adj_output(adj_output.vector())

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        output = vector_boundary_to_mesh(tlm_input, W.mesh())
        self.get_outputs()[0].add_tlm_output(output)

    @no_annotations
    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        mesh = self.get_dependencies()[1].output
        hessian_value = backend.Function(W)
        hessian_value.vector()[:] = hessian_input
        hessian_output = vector_mesh_to_boundary(hessian_value, mesh)
        self.get_dependencies()[0].add_hessian_output(hessian_output.vector())

    @no_annotations
    def recompute(self):
        deps = self.get_dependencies()
        W = self.get_outputs()[0].output.function_space()
        deps[0].saved_output.set_allow_extrapolation(True)
        output = vector_boundary_to_mesh(deps[0].saved_output, W.mesh())
        self.get_outputs()[0].checkpoint = output


def transfer_to_boundary(*args, **kwargs):
    """
    Transfers values from a CG1 function on a mesh to its corresponding
    BoundaryMesh.
    """
    annotate = annotate_tape(kwargs)
    with stop_annotating():
        output = vector_mesh_to_boundary(*args)
    output = create_overloaded_object(output)

    if annotate:
        block = VolumeTransferBlock(args[0], args[1])
        tape = get_working_tape()
        tape.add_block(block)
        block.add_output(output.block_variable)

    return output


class VolumeTransferBlock(Block):
    def __init__(self, func, other):
        super(VolumeTransferBlock, self).__init__()
        self.add_dependency(func)
        self.add_dependency(func.function_space().mesh())

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        b_mesh = self.get_dependencies()[1].output
        adj_value = backend.Function(W, adj_input)
        adj_output = vector_boundary_to_mesh(adj_value, b_mesh)
        self.get_dependencies()[0].add_adj_output(adj_output.vector())

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        output = vector_mesh_to_boundary(tlm_input, W.mesh())
        self.get_outputs()[0].add_tlm_output(output)

    @no_annotations
    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        mesh = self.get_dependencies()[1].output
        hessian_value = backend.Function(W, hessian_input)
        hessian_output = vector_boundary_to_mesh(hessian_value, mesh)
        self.get_dependencies()[0].add_hessian_output(hessian_output.vector())

    @no_annotations
    def recompute(self):
        deps = self.get_dependencies()
        W = self.get_outputs()[0].output.function_space()
        deps[0].saved_output.set_allow_extrapolation(True)
        output = vector_mesh_to_boundary(deps[0].saved_output, W.mesh())
        self.get_outputs()[0].checkpoint = output
