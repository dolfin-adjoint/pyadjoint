from dolfin import *
from mshr import *

rect = Rectangle(Point(0, 0), Point(30, 10))
circ = Circle(Point(10, 5), 2.5)
domain = rect - circ
N = 50

mesh = generate_mesh(domain, N)

filename_xdmf = XDMFFile(mpi_comm_world(), "rectangle-less-circle.xdmf")
filename_xdmf.write(mesh)
