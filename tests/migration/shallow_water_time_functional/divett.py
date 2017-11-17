from dolfin import *
from math import exp, sqrt, pi

import sw_lib

params=sw_lib.parameters({
    'depth' : 2.,
    'g' : 9.81,
    'f' : 0.0,
    'dump_period' : 1,
    'eta0' : 2 # Wave height
    })

# Basin radius.
basin_x=3000 # The length of the basin
basin_y=1000 # The width of the basin
nx=20 # Number of cells in x direction
ny=3 # Number of cells in y direction
# Long wave celerity.
c=sqrt(params["g"]*params["depth"])


params["finish_time"]=100
params["dt"]=params["finish_time"]/4000.

class InitialConditions(Expression):
    def eval(self, values, X):
        values[0]=params['eta0']*sqrt(params['g']*params['depth'])*cos(pi*X[0]/3000)
        values[1]=0.
        values[2]=params['eta0']*cos(pi*X[0]/3000)
    def value_shape(self):
        return (3,)


mesh = RectangleMesh(mpi_comm_world(), Point(0, 0), Point(basin_x, basin_y), nx, ny)
mesh.order()
mesh.init()

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], basin_x)

class Sides(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], basin_y))

# Initialize sub-domain instances
left = Left()
right = Right()
sides = Sides()

# Initialize mesh function for boundary domains
try:
    boundaries = FacetFunction("sizet", mesh)
except:
    boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
sides.mark(boundaries, 3)
ds = Measure("ds")(subdomain_data=boundaries)
