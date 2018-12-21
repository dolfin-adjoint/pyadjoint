from dolfin import *
from dolfin_adjoint import *
from pyadjoint.tape import stop_annotating
from numpy import isclose

def test_read_xdmf_mesh():
    mesh_2d = Mesh()
    with XDMFFile("mesh_2d.xdmf") as infile:
        infile.read(mesh_2d)
    mvc = MeshValueCollection("size_t", mesh_2d, 1)
    with XDMFFile("mvc_1d.xdmf") as infile:
        infile.read(mvc, "name_to_read")

def test_read_checkpoint():
    with stop_annotating():
        mesh = UnitSquareMesh(10,10)
        V = FunctionSpace(mesh, "CG", 1)
        x = SpatialCoordinate(mesh)
        v = project(x[0]*x[1]*cos(x[1]), V)
        out = XDMFFile("scalar.xdmf")
        out.write_checkpoint(v, "u", 0.0)
        out.close()

    mesh = UnitSquareMesh(10,10)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)
    c = Control(v)
    J = assemble(v*dx)
    infile = XDMFFile("scalar.xdmf")
    u = Function(V)
    infile.read_checkpoint(u,'u', -1)
    infile.close()
    J += assemble(u*dx)
    Jhat = ReducedFunctional(J, c)
    with stop_annotating():
        v = interpolate(Expression("x[0]*x[1]",degree=1), V)
        assert(0.1908866453380181120 + 0.25, Jhat(v))
