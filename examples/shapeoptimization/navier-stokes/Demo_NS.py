from dolfin import *
from dolfin_adjoint import *
set_log_level(LogLevel.ERROR)

# Create mesh and facet-function from gmsh file.
mesh = Mesh("meshes/mesh.xml")
Vol0 = 1-assemble(1*dx(domain=mesh))

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S)
ALE.move(mesh, s)

facet_function = MeshFunction("size_t", mesh, "meshes/mesh_facet_region.xml")
inlet, outlet, walls, obstacle = 1, 2, 3, 4 # Boundary Markers

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

dt = 0.01
T = 1
nu = Constant(0.01, name="Viscosity")

p_in = Expression("sin(t*T)", t=0.0, T=T, degree=2)
u_walls = Constant((0.0,0.0), name="Non-slip")
u_obstacle = Constant((0.0,0.0), name="Object")

bc_walls = DirichletBC(V, u_walls, facet_function, walls)
bc_obstacle = DirichletBC(V, u_obstacle, facet_function, obstacle)
bc_inlet = DirichletBC(Q, p_in, facet_function, inlet)
bc_outlet = DirichletBC(Q, 0, facet_function, outlet)
bcu = [bc_walls, bc_obstacle]
bcp = [bc_inlet, bc_outlet]

u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

k = Constant(dt, name="dt")
f = Constant((0,0), name="f")

# Tentative velocity step
F1 = (1/k)*inner(u-u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
     nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/k)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True
# Create files for storing solution
ufile = File("output/velocity.pvd")
pfile = File("output/pressure.pvd")

# Time-stepping
t = dt
times = []
J = 0
while t < T + DOLFIN_EPS:
    print("Time: %.2e" %t)
    # Update pressure boundary condition
    p_in.t = t

    # Compute tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")

    # Pressure correction
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", "default")

    # Velocity correction
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")

    # Save to file
    ufile << u1
    pfile << p1

    # Move to next time step
    u0.assign(u1, annotate=True)
    t += dt
    times.append(float(t))
    J += dt*assemble(0.5*nu*inner(grad(u1),grad(u1))*dx)


Vol = 1-assemble(1*dx(domain=mesh))
x = SpatialCoordinate(mesh)
bx = (0.5-assemble(x[0]*dx))/Vol
by = (0.5-assemble(x[1]*dx))/Vol
J+= (Vol-Vol0)**2
J+= ((bx-0.5)**2 + (by-0.5)**2)
Jhat = ReducedFunctional(J, Control(s))

# Visualize pyadjoint tape
Jhat.optimize()
tape.visualise("output/tape_ns.dot", dot=True)

def riesz_representation(integral, alpha=10):
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
meshout = File("output/mesh_ns.pvd")
meshout << mesh

while it <= max_it and red > red_tol:
    # Compute derivative of previous configuration
    dJdm = Jhat.derivative(options={"riesz_representation":
                                    riesz_representation})

    # Linesearch
    step = 1
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
