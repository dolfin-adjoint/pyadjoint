import pygmsh
res = 0.025
geometry = pygmsh.built_in.Geometry()

# Create an obstacle through bsplines
p1 = geometry.add_point([0.25,0.5,0], res/2)
p2 = geometry.add_point([0.5,0.6,0], res/2)
p3 = geometry.add_point([0.9,0.5,0], res/2)
p4 = geometry.add_point([0.5,0.4,0], res/2)
obstacle_top = geometry.add_bspline([p3,p2,p1,p4,p3])
loop = geometry.add_line_loop([obstacle_top])
rectangle = geometry.add_rectangle(0,1,0,1,0, res, holes=[loop])
geometry.add_physical_surface(rectangle.surface,label=5)
geometry.add_physical_line(loop.lines, label=4)
geometry.add_physical_line(rectangle.line_loop.lines[3], label=1) # inlet
geometry.add_physical_line([rectangle.line_loop.lines[0],
                            rectangle.line_loop.lines[2]], label=3) # walls
geometry.add_physical_line(rectangle.line_loop.lines[1], label=2) # outlet

mesh_data = pygmsh.generate_mesh(geometry, geo_filename="mesh.geo")
import os;
os.system("gmsh -3 mesh.geo")
os.system("dolfin-convert mesh.msh mesh.xml")
os.system("mkdir -p meshes")
os.system("mv mesh* meshes/")
