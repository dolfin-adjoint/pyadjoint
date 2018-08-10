import pygmsh
res = 0.05
geometry = pygmsh.built_in.Geometry()
obstacle = geometry.add_rectangle(0.26,0.6,0.36,0.48,0, res)
rectangle = geometry.add_rectangle(0,1,0,1,0, res, holes=[obstacle])

geometry.add_physical_surface(rectangle.surface,label=1)
geometry.add_physical_line(obstacle.line_loop.lines, label=2)
geometry.add_physical_line(rectangle.line_loop.lines, label=3)

mesh_data = pygmsh.generate_mesh(geometry, geo_filename="mesh.geo")
import os;
os.system("gmsh -3 mesh.geo")
os.system("dolfin-convert mesh.msh mesh.xml")
os.system("mkdir -p meshes")
os.system("mv mesh* meshes/")
