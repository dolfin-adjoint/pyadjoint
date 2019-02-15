from pygmsh import generate_mesh
from pygmsh.built_in.geometry import Geometry
import meshio

inflow = 1
outflow = 2
walls = 3
obstacle = 4
L = 1 # Length of channel
H = 1 # Width of channel
c_x,c_y  = L/2, H/2 # Position of object
r_x = 0.126157 # Radius of object



def single_mesh(res=0.025):
    """ 
    Creates a single mesh containing a circular obstacle
    """

    geometry = Geometry()
    c = geometry.add_point((c_x,c_y,0))

    # Elliptic obstacle
    p1 = geometry.add_point((c_x-r_x, c_y,0))
    p2 = geometry.add_point((c_x, c_y+r_x,0))
    p3 = geometry.add_point((c_x+r_x, c_y,0))
    p4 = geometry.add_point((c_x, c_y-r_x,0))
    arc_1 = geometry.add_ellipse_arc(p1, c, p2, p2)
    arc_2 = geometry.add_ellipse_arc(p2, c, p3, p3)
    arc_3 = geometry.add_ellipse_arc(p3, c, p4, p4)
    arc_4 = geometry.add_ellipse_arc(p4, c, p1, p1)
    obstacle_loop = geometry.add_line_loop([arc_1, arc_2, arc_3, arc_4])

    rectangle = geometry.add_rectangle(0,L,0,H,0, res, holes=[obstacle_loop])
    flow_list = [rectangle.line_loop.lines[3]]
    obstacle_list = obstacle_loop.lines
    walls_list = [rectangle.line_loop.lines[0], rectangle.line_loop.lines[2]]
    geometry.add_physical_surface(rectangle.surface,label=12)
    geometry.add_physical_line(flow_list, label=inflow)
    geometry.add_physical_line(walls_list, label=walls)
    geometry.add_physical_line([rectangle.line_loop.lines[1]], label=outflow)
    geometry.add_physical_line(obstacle_list, label=obstacle)
    field = geometry.add_boundary_layer(edges_list=obstacle_loop.lines,
                                        hfar=res, hwall_n=res/8, thickness=res/2)
    geometry.add_background_field([field])

    (points, cells, point_data,
     cell_data, field_data) = generate_mesh(geometry, prune_z_0=True,
                                            geo_filename="test.geo")
    
    meshio.write("mesh.xdmf", meshio.Mesh(
        points=points, cells={"triangle": cells["triangle"]}))
    
    meshio.write("mf.xdmf", meshio.Mesh(
        points=points, cells={"line": cells["line"]},
        cell_data={"line": {"name_to_read":
                            cell_data["line"]["gmsh:physical"]}}))

        
if __name__=="__main__":
    import sys
    try:
        res = float(sys.argv[1])
    except IndexError:
        res = 0.05
    single_mesh(res)
