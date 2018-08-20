from dolfin import *
from dolfin_adjoint import *
set_log_level(LogLevel.ERROR)
# Create mesh and facet-function from gmsh file.
mesh = Mesh("meshes/mesh.xml")
Vol0 = 1-assemble(1*dx(domain=mesh))

facet_function = MeshFunction("size_t", mesh, "meshes/mesh_facet_region.xml")
inlet, outlet, walls, obstacle = 1, 2, 3, 4 # Boundary Markers

# Create the Mixed-function space and corresponding test and trial functions
V_h = VectorElement("CG", mesh.ufl_cell(), 2)
Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V_h * Q_h)
v, q = TestFunctions(W)
up = Function(W, name="w")
u, p = split(up)

# Move mesh
S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Deformation")
ALE.move(mesh, s) 

# Physical parameters and boundary data
nu = Constant(5e-2, name="Viscosity")
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
UFile = XDMFFile("output/u_ns.xdmf")
PFile = XDMFFile("output/p_ns.xdmf")
UFile.write(up.split()[0])
PFile.write(up.split()[1])


# Creating Reduced Functional
J = assemble(0.5 * nu * inner(grad(u), grad(u))*dx)
Vol = 1-assemble(1*dx(domain=mesh))
x = SpatialCoordinate(mesh)
bx = (0.5-assemble(x[0]*dx))/Vol
by = (0.5-assemble(x[1]*dx))/Vol
J+= 7.5e1*(Vol-Vol0)**2
J+= 7.5e1*((bx-0.5)**2 + (by-0.5)**2)
c = Control(s)
Jhat = ReducedFunctional(J, c)

# Visualize pyadjoint tape
Jhat.optimize()
tape.visualise("output/tape_ns_timeindep.dot", dot=True)


def riesz_representation(integral, alpha=5):
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


it, max_it = 1, 250
min_stp = 1e-6 
red_tol = 1e-6
red = 2*red_tol

move = Function(S)
move_step = Function(S)
Js = [Jhat(move)]
meshout = File("output/mesh.pvd")
meshout << mesh

while it <= max_it and red > red_tol:
    # Compute derivative of previous configuration
    dJdm = Jhat.derivative(options={"riesz_representation":
                                    riesz_representation})
    # Linesearch
    step = 2.5
    while step > min_stp:
        # Evaluate functional at new step 
        move_step.vector()[:] = move.vector() - step*dJdm.vector()
        J_step = Jhat(move_step)

        # Check if functional is decreasing
        if J_step - Js[-1] < 0:
            break
        else:
            # Reset mesh and half step-size
            step /= 2
            if step <= min_stp:
                raise(ValueError("Minimum step-length reached"))

    move.assign(move_step)
    Js.append(J_step)
    meshout << mesh
    red = abs((Js[-1] - Js[-2])/Js[-1])
    it += 1
    print("Iteration: %d, Rel. Red.: %.2e" %(it-1, red))


print("Total number of iterations: %d" % (it-1))
print("-"*5, "Optimization Finished", "-"*5)
print("Initial Functional value: %.2f" % Js[0])
print("Final Functional value: %.2f" % Js[-1])


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
Jhat(s0)
s.tlm_value = deform
tape.evaluate_tlm()
dJdm = Jhat.derivative().vector().inner(deform.vector())
Hm = compute_hessian(J, c, deform).vector().inner(deform.vector())
taylor_test(Jhat, s0, deform, dJdm=dJdm, Hm=Hm)



