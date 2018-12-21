import backend
import numpy
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.block import Block
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations


def SyncSum(LocValues):
    if backend.MPI.size(backend.MPI.comm_world) > 1:
        from mpi4py import MPI as MPI4PY
        comm = MPI4PY.COMM_WORLD
        MPI4PYSize = comm.Get_size()
        Shape = LocValues.shape
        if len(Shape) > 1:
            LocValues = LocValues.reshape(LocValues.size)
        MySize = int(len(LocValues))
        NormalsAllProcs = numpy.zeros(MPI4PYSize*MySize, dtype=LocValues.dtype)
        comm.Allgather(LocValues, NormalsAllProcs)
        # MPI.gather(MPI.comm_world, LocValues, NormalsAllProcs)
        for i in range(MySize):
            LocValues[i] = 0.0
            for j in range(MPI4PYSize):
                LocValues[i] = LocValues[i] + NormalsAllProcs[MySize*j + i]
        if len(Shape) > 1:
            LocValues = LocValues.reshape(Shape)
        return LocValues
    else:
        return LocValues

def ReduceFunctionToSurface(vector, boundarymesh):
    """
    Reduces a CG-1 function from a mesh to a CG-1 function on the boundary 
    mesh
    """
    mesh = vector.function_space().mesh()
    MaxDim = mesh.geometric_dimension()
    surface_space = backend.VectorFunctionSpace(boundarymesh, "CG", 1)
    surfacevector = backend.Function(surface_space)
    (sdmin, sdmax) = surfacevector.vector().local_range()
    sdmin = int(sdmin/MaxDim)
    sdmax = int(sdmax/MaxDim)
    LocValues = numpy.zeros(MaxDim*(sdmax-sdmin))

    VGlobal = numpy.zeros(len(vector.vector()))
    (vdmin, vdmax) = vector.vector().local_range()
    vdmin = int(vdmin/MaxDim)
    vdmax = int(vdmax/MaxDim)
    # OwnerRange = vector.function_space().dofmap().ownership_range()
    DofToVert = backend.dof_to_vertex_map(backend.FunctionSpace(mesh, "CG", 1))
    for i in range(vdmax-vdmin):
        Vert = backend.MeshEntity(mesh, 0, DofToVert[i])
        GlobalIndex = Vert.global_index()
        # IsOwned = OwnerRange[0] <= GlobalIndex and GlobalIndex<=OwnerRange[1]
        # if IsOwned:
        for j in range(MaxDim):
            value = vector.vector()[MaxDim*(i+vdmin)+j]
            VGlobal[MaxDim*GlobalIndex+j] = value
    VGlobal = SyncSum(VGlobal)
    mapa = boundarymesh.entity_map(0)
    DofToVert = backend.dof_to_vertex_map(backend.FunctionSpace(boundarymesh, "CG", 1))
    for i in range(sdmax-sdmin):
        VolVert = backend.MeshEntity(mesh, 0, mapa[int(DofToVert[i])])
        GlobalIndex = VolVert.global_index()
        for j in range(MaxDim):
            value = VGlobal[MaxDim*GlobalIndex+j]
            LocValues[MaxDim*i+j] = value

    surfacevector.vector().set_local(LocValues)
    surfacevector.vector().apply('')
    return surfacevector

def InjectFunctionFromSurface(f, mesh):
    """ Take a CG1 function f defined on a surface mesh and return a 
    volume vector with same values on boundary but zero in volume
    """
    MaxDim = mesh.geometric_dimension()

    boundarymesh = f.function_space().mesh()
    SpaceS = backend.FunctionSpace(mesh, "CG", 1)
    SpaceV = backend.VectorFunctionSpace(mesh, "CG", 1, MaxDim)
    F = backend.Function(SpaceV)
    LocValues = numpy.zeros(F.vector().local_size())
    map = boundarymesh.entity_map(0)
    OwnerRange = SpaceV.dofmap().ownership_range()
    d2v = backend.dof_to_vertex_map(backend.FunctionSpace(boundarymesh, "CG", 1))
    v2d = backend.vertex_to_dof_map(SpaceS)
    for i in range(int(f.vector().local_size()/MaxDim)):

        GVertID = backend.Vertex(boundarymesh, d2v[i]).index()
        PVertID = map[GVertID]
        PDof = v2d[PVertID]
        l_2_g_index = SpaceV.dofmap().local_to_global_index(PDof)
        IsOwned = OwnerRange[0]/MaxDim <= l_2_g_index and l_2_g_index<=OwnerRange[1]/MaxDim
        if IsOwned:
            for j in range(MaxDim):
                value = f.vector()[MaxDim*i+j]
                LocValues[PDof*MaxDim+j] = value
    F.vector().set_local(LocValues)
    F.vector().apply("")
    return F


def transfer_from_boundary(*args, **kwargs):
    """
    Transfers values from a CG1 function on the BoundaryMesh to its
    original mesh
    """
    annotate = annotate_tape(kwargs)
    with stop_annotating():
        output = InjectFunctionFromSurface(*args)
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
        self.add_dependency(func.block_variable)
        self.add_dependency(func.function_space().mesh().block_variable)

    @no_annotations
    def evaluate_adj(self, markings=False):
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        b_mesh = self.get_dependencies()[1].output
        adj_value = backend.Function(W, adj_input)
        adj_output = ReduceFunctionToSurface(adj_value, b_mesh)
        self.get_dependencies()[0].add_adj_output(adj_output.vector())

    @no_annotations
    def evaluate_tlm(self, markings=False):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        output = InjectFunctionFromSurface(tlm_input, W.mesh())
        self.get_outputs()[0].add_tlm_output(output)

    @no_annotations
    def evaluate_hessian(self, markings=False):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        W = self.get_outputs()[0].output.function_space()
        mesh = self.get_dependencies()[1].output
        hessian_value = backend.Function(W, hessian_input)
        hessian_output = ReduceFunctionToSurface(hessian_value, mesh)
        self.get_dependencies()[0].add_hessian_output(hessian_output.vector())


    @no_annotations
    def recompute(self):
        deps = self.get_dependencies()
        W = self.get_outputs()[0].output.function_space()
        deps[0].saved_output.set_allow_extrapolation(True)
        output = InjectFunctionFromSurface(deps[0].saved_output, W.mesh())
        self.get_outputs()[0].checkpoint = output
