"""
Make a figure-8 Klein bottle
Parametric description taken from

http://paulbourke.net/geometry/klein/
"""

from __future__ import print_function
from dolfin import *
import numpy

n = 256
mesh = RectangleMesh(Point(0, 0), Point(4*pi, 2*pi), n, n)

# First step: "wrap" the boundaries (right-to-left, top-to-bottom)
# so that when we transform to the Klein bottle, the edges "join up"
# Sub domain for Periodic boundary condition
class LeftRightPeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(abs(x[0]) < DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 4*pi
        y[1] = x[1]

class TopBottomPeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        #print "inside(%s)? %s" % (x, bool(abs(x[1] + pi) < DOLFIN_EPS and on_boundary))
        return bool(abs(x[1]) < DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - 2*pi
        #print "map(%s): %s" % (x, y)

def wrap_mesh(mesh, pbs):
    print("Wrapping input mesh: # vertices == ", mesh.num_vertices())

    def merge_slave_pairs(merged, new_map):
        for key in merged:
            if key in new_map:
                del new_map[key]

            if merged[key] in new_map:
                merged[key] = new_map[merged[key]][1]

        copy = dict(merged)
        for key in copy:
            if merged[key] == key:
                del merged[key]

        for key in new_map:
            merged[key] = new_map[key][1]

    merged_slave_pairs = {}
    vertex_map = {}

    for pb in pbs:
        pbc = PeriodicBoundaryComputation()
        slave_pairs = pbc.compute_periodic_pairs(mesh, pb, 0)
        assert len(slave_pairs) > 0, "Periodic boundary found nothing"
        merge_slave_pairs(merged_slave_pairs, slave_pairs)
        del slave_pairs

    slaves_hit = 0
    vertex_map = {}
    for i in range(mesh.num_vertices()):
        if i in merged_slave_pairs:
            # got a slave vertex, increment the counter
            slaves_hit += 1
        else:
            vertex_map[i] = i - slaves_hit

    for i in merged_slave_pairs:
        vertex_map[i] = vertex_map[merged_slave_pairs[i]]

    wrapped_mesh = Mesh()
    editor = MeshEditor()
    editor.open(wrapped_mesh, mesh.topology().dim(), mesh.geometry().dim())
    editor.init_vertices(mesh.num_vertices() - slaves_hit)
    editor.init_cells(mesh.num_cells())

    coords = mesh.coordinates()

    for c in cells(mesh):
        editor.add_cell(c.index(), numpy.array([vertex_map[v.index()] for v in vertices(c)], dtype="uintp"))

    for (i, v) in enumerate(vertices(mesh)):
        if i not in merged_slave_pairs:
            editor.add_vertex(vertex_map[i], coords[i])

    editor.close()
    print("Wrapped output mesh: # vertices == ", wrapped_mesh.num_vertices())
    return wrapped_mesh

wrapped_mesh = wrap_mesh(mesh, [LeftRightPeriodicBoundary(), TopBottomPeriodicBoundary()])

# Map parametric coordinates (u, v) to (x, y, z)
a = 2.0 ; n = 2; m = 1
code = r'''
class KleinMap : public Expression
{
  public:

void eval(Array<double>& values, const Array<double>& x) const
{
  double u = x[0];
  double v = x[1];
  values[0] = (%(a)s + cos(%(n)s*u/2.0) * sin(v) - sin(%(n)s*u/2.0) * sin(2*v)) * cos(%(m)s*u/2.0);
  values[1] = (%(a)s + cos(%(n)s*u/2.0) * sin(v) - sin(%(n)s*u/2.0) * sin(2*v)) * sin(%(m)s*u/2.0);
  values[2] = sin(%(n)s*u/2.0) * sin(v) + cos(%(n)s*u/2.0) * sin(2*v);
}

std::size_t value_rank() const
{
  return (std::size_t) 1;
}

std::size_t value_dimension(std::size_t i) const
{
  return (std::size_t) 3;
}

};
''' % {'a': a, 'm': m, 'n': n}

V = VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

KleinMap = Expression(code, element=V.ufl_element())

def transform_mesh(mesh, coord_map):
    if isinstance(coord_map, Expression):
        single_map = coord_map
        coord_map = lambda coords: [single_map(pt) for pt in coords]

    mapped_coords = coord_map(mesh.coordinates())
    assert len(mapped_coords[0].shape) == 1
    gdim = mapped_coords[0].shape[0]
    assert gdim >= mesh.geometry().dim()
    tdim = mesh.topology().dim()

    new_mesh = Mesh()
    editor = MeshEditor()
    editor.open(new_mesh, tdim, gdim)
    editor.init_vertices(mesh.num_vertices())
    editor.init_cells(mesh.num_cells())

    for c in cells(mesh):
        editor.add_cell(c.index(), numpy.array([v.index() for v in vertices(c)], dtype="uintp"))

    for (i, v) in enumerate(vertices(mesh)):
        editor.add_vertex(i, mapped_coords[i])

    editor.close()
    return new_mesh

new_mesh = transform_mesh(wrapped_mesh, KleinMap)
outfile = XDMFFile(mpi_comm_world(), 'klein.xdmf')
outfile.write(new_mesh)

plot(new_mesh, wireframe=True, interactive=True)
