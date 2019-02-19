import os

from dolfin import *
from dolfin_adjoint import *
from pyadjoint.tape import stop_annotating
from numpy import isclose

def file_from_curr_dir(filename):
    return os.path.dirname(__file__) + "/" + filename

def test_read_xdmf_mesh():
    mesh_2d = Mesh()
    with XDMFFile(file_from_curr_dir("mesh_2d.xdmf")) as infile:
        infile.read(mesh_2d)
    mvc = MeshValueCollection("size_t", mesh_2d, 1)
    with XDMFFile(file_from_curr_dir("mvc_1d.xdmf")) as infile:
        infile.read(mvc, "name_to_read")

def test_read_checkpoint():
    with stop_annotating():
        N = 15
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh, "CG", 1)
        x = SpatialCoordinate(mesh)
        v = project(x[0]*x[1]*cos(x[1]), V)
        out = XDMFFile(file_from_curr_dir("scalar.xdmf"))
        out.write_checkpoint(v, "u", 0.0)
        out.close()
        exact = assemble(v*dx)

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)
    c = Control(v)
    J = assemble(v*dx)
    infile = XDMFFile(file_from_curr_dir("scalar.xdmf"))
    u = Function(V)
    infile.read_checkpoint(u,'u', -1)
    infile.close()
    J += assemble(u*dx)
    Jhat = ReducedFunctional(J, c)
    with stop_annotating():
        x = SpatialCoordinate(mesh)
        v = project(x[0]*x[1], V)
        assert(isclose(exact + 0.25, Jhat(v)))
