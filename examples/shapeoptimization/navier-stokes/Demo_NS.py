from dolfin import *
from dolfin_adjoint import *
set_log_level(LogLevel.ERROR)

parameters["krylov_solver"]["relative_tolerance"] = 1e-14

# Create mesh and facet-function from gmsh file.
mesh = Mesh("meshes/mesh.xml")
X0 = SpatialCoordinate(mesh)

# Compute Volume and Barycenter of obstacle assuming a rectangular duct
# FIXME: Use mesh coordinates, dolfin adjoints fails in specification
coords_copy = mesh.coordinates().copy()
facet_function = MeshFunction("size_t", mesh, "meshes/mesh_facet_region.xml")
inlet, outlet, walls, obstacle = 1, 2, 3, 4 # Boundary Markers

def forward(s):
    global facet_function
    mesh.coordinates()[:] = coords_copy
    # FIXME: Use mesh coordinates, dolfin adjoints fails in specification
    L = 2  # max(coords[:,0])-min(coords[:,0])
    H = 1  # max(coords[:,1])-min(coords[:,1])
    V_fluid = assemble(1 * dx(domain=mesh))
    Bx0 = (L**2*H/2-assemble(X0[0]*dx))/(L*H- V_fluid)
    By0 = (L*H**2/2-assemble(X0[1]*dx))/(L*H- V_fluid)
    Vol0 = L * H - V_fluid
    ALE.move(mesh, s)
    facet_function = MeshFunction("size_t", mesh, "meshes/mesh_facet_region.xml")

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    dt =0.1
    T = 1
    nu = Constant(0.001, name="Viscosity")

    p_in = Expression("0.5*sin(pi/T*t)", t=0.0, T=T, degree=2)
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
        J += assemble(0.5*nu*inner(grad(u1),grad(u1))*dx)

    # Quadratic pentaltization of volume and barycenter offset
    alpha_p = 1e1
    x = SpatialCoordinate(mesh)
    Vol = L*H - assemble(1*dx(domain=mesh))
    J += alpha_p*(Vol-Vol0)**2
    Bx = ((L**2*H/2-assemble(x[0]*dx(domain=mesh)))/
          (L*H -assemble(1*dx(domain=mesh))))
    By = ((L*H**2/2-assemble(x[1]*dx(domain=mesh)))/
          (L*H - assemble(1*dx(domain=mesh))) )
    J += alpha_p*((Bx-Bx0)**2 + (By-By0)**2)

    return J

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="Deformation")
J = forward(s)
c = Control(s)
Jhat = ReducedFunctional(J, c)

# Visualize pyadjoint tape
Jhat.optimize()
#tape.visualise("output/tape_ns.dot", dot=True)

def riesz_representation(integral, alpha=10, taylor = False):
    """
    Returns a smoothed 'H1'-representation of an integral.
    Note that the boundary values are set strongly. This is due
    to the fact that we would like to keep the movement normal to the boundary
    """
    # Remove internal movement from the representation
    # boundary_rep = Function(S)
    if not taylor:
        tmp_bc = DirichletBC(S, Constant((0.0,0.0)), "on_boundary")
        vol = integral.copy()
        volume = tmp_bc.apply(vol)
        integral = integral-vol

    # Use a smoothed h1 riesz representation
    u, v = TrialFunction(S), TestFunction(S)
    a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
    A = assemble(a)
    bcs = [DirichletBC(S, Constant((0.0, 0.0)), facet_function, walls),
           DirichletBC(S, Constant((0.0, 0.0)), facet_function, inlet),
           DirichletBC(S, Constant((0.0, 0.0)), facet_function, outlet)]
    # integral = assemble(inner(boundary_rep, v)*ds)
    [bc.apply(A, integral) for bc in bcs]
    representation = Function(S)
    solve(A, representation.vector(), integral)
    # import matplotlib.pyplot as plt
    # plot(representation)
    # plt.show()
    return representation

# it, max_it = 1, 100
# min_stp = 1e-9
# red_tol = 1e-6
# red = 2*red_tol

# move = Function(S)
# move_step = Function(S)
# Js = [Jhat(move)]
# meshout = File("output/mesh_ns.pvd")
# meshout << mesh

# while it <= max_it and red > red_tol:
#     # Compute derivative of previous configuration
#     dJdm = Jhat.derivative(options={"riesz_representation":
#                                     riesz_representation})

#     # Linesearch
#     step = 100
#     while step > min_stp:
#         # Evaluate functional at new step 
#         move_step.vector()[:] = move.vector() - step*dJdm.vector()
#         J_step = Jhat(move_step)

#         # Check if functional is decreasing
#         if J_step - Js[-1] < 0:
#             break
#         else:
#             # Reset mesh and half step-size
#             step /= 2
#             if step <= min_stp:
#                 raise(ValueError("Minimum step-length reached"))
#     print("step: %.2e" %step)
#     move.assign(move_step)
#     Js.append(J_step)
#     meshout << mesh
#     red = abs((Js[-1] - Js[-2])/Js[-1])
#     it += 1
#     print("Iteration: %d, Rel. Red.: %.2e" %(it-1, red))


# print("Total number of iterations: %d" % (it-1))
# print("-"*5, "Optimization Finished", "-"*5)
# print("Initial Functional value: %.2f" % Js[0])
# print("Final Functional value: %.2f" % Js[-1])

# Set taylor-test in normal direcction
from femorph import VolumeNormal
n_vol = VolumeNormal(mesh)
deform_backend = Function(S)
deform_backend.vector()[:] = 2*n_vol.vector()
r = assemble(inner(deform_backend, TestFunction(S))*ds)
deform = riesz_representation(r, taylor=True)

s0 = Function(S)
from pyadjoint.tape import stop_annotating
# 0th order taylor test, expected convergence rate 1
with stop_annotating():
    taylor_test(Jhat, s0, deform, dJdm=0)
# Reset mesh deformation
Jhat(s0)
# 1st order taylor test, expected convergence rate 2
with stop_annotating():
    taylor_test(Jhat, s0, deform)
Jhat(s0)

# 2nd order taylor test, expected convergence rate 3
s.tlm_value = deform
tape.evaluate_tlm()
dJdm = Jhat.derivative().vector().inner(deform.vector())
Hm = compute_hessian(J, c, deform).vector().inner(deform.vector())
with stop_annotating():
    taylor_test(Jhat, s0, deform, dJdm=dJdm, Hm=Hm)

# 2nd order taylor test of forward-function, expected convergence 3
with stop_annotating():
    taylor_test(forward, s0, deform, dJdm=dJdm, Hm=Hm)
