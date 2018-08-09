from dolfin import *
from dolfin_adjoint import *
from os import system, chdir

# Create mesh and facet-function from gmsh file.
chdir("mesh")
system("gmsh -2 mesh_stokes.geo")
system("dolfin-convert mesh_stokes.msh mesh_stokes.xml")
chdir("..")
mesh = Mesh("mesh/mesh_stokes.xml")
facet_function = MeshFunction("size_t", mesh, "mesh/mesh_stokes_facet_region.xml")
inlet, outlet, walls, obstacle = 1, 2, 3, 4 # Boundary Markers

# Create the Mixed-function space and corresponding test and trial functions
V_h = VectorElement("CG", mesh.ufl_cell(), 2)
Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V_h * Q_h)
v, q = TestFunctions(W)
up = Function(W)
u, p = split(up)

# Move mesh
S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Deformation")
ALE.move(mesh, s) 

# Physical parameters and boundary data
nu = Constant(0.01, name="Viscosity")
u_inlet = Function(W.sub(0).collapse())
u_inlet.interpolate(Expression(("sin(pi*x[1])", "0"), degree=1,
                                 name="Inlet Velocity",
                                 domain=mesh))
u_inlet.rename("u_inlet", "")
u_walls = Constant((0.0,0.0), name="No-slip walls")
u_obstacle = Constant((0.0,0.0), name="No-slip obstacle")

F = (nu * inner(grad(u), grad(v))*dx + inner(dot(grad(u),u), v)*dx 
     - inner(p, div(v)) * dx - inner(q, div(u)) * dx)

bc_inlet = DirichletBC(W.sub(0), u_inlet, facet_function, inlet)
bc_walls = DirichletBC(W.sub(0), u_walls, facet_function, walls)
bc_obstacle = DirichletBC(W.sub(0), u_obstacle, facet_function, obstacle)
bcs = [bc_inlet, bc_walls, bc_obstacle]

# Solve the Mixed-problem
solve(F==0, up, bcs=bcs)

# Save output values to file
UFile = XDMFFile("output/u_stokes.xdmf")
PFile = XDMFFile("output/p_stokes.xdmf")
UFile.write(up.split()[0])
PFile.write(up.split()[1])
# u_, p_ = up.split()
# u_.rename("Velocity", "")
# p_.rename("Pressure", "")


# Creating Reduced Functional
J = assemble(0.5 * nu * inner(grad(u), grad(u))*dx)
Jhat = ReducedFunctional(J, Control(s))
Jhat.optimize()

def riesz_representation(integral, alpha=1):
    """
    Returns a smoothed 'H1'-representation of an integral.
    Note that the boundary values are set strongly. This is due
    to the fact that we would like to keep the movement normal to the boundary
    """
    u, v = TrialFunction(S), TestFunction(S)
    a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
    A = assemble(a)
    bcs = [DirichletBC(S, Constant((0.0, 0.0)), facet_function, walls),
           DirichletBC(S, Constant((0.0, 0.0)), facet_function, inlet),
           DirichletBC(S, Constant((0.0, 0.0)), facet_function, outlet)]
    [bc.apply(A, integral) for bc in bcs]
    representation = Function(S)
    solve(A, representation.vector(), integral)
    return representation

# Set taylor-test in steepest accent direction
from femorph import VolumeNormal
n_vol = VolumeNormal(mesh)
deform_backend = Function(S)
deform_backend.vector()[:] = 2*n_vol.vector()
rhs = assemble(inner(deform_backend, TestFunction(S))*ds)
deform = riesz_representation(rhs)



s0 = Function(S)
taylor_test(Jhat, s0, deform, dJdm=0)
taylor_test(Jhat, s0, deform)

# Visualize pyadjoint tape
tape.visualise("output/tape_stokes.dot", dot=True)


