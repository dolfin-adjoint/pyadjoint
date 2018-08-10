from dolfin import *
from dolfin_adjoint import *
from ufl import replace
import matplotlib.pyplot as plt
set_log_level(LogLevel.ERROR)

mesh = Mesh("meshes/mesh.xml")
mf = MeshFunction("size_t", mesh, "meshes/mesh_facet_region.xml")
obstacle = 2
boundary = 3

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="s")
ALE.move(mesh, s)

# Setup
V = FunctionSpace(mesh, "CG", 1)
u0, v = TrialFunction(V), TestFunction(V)
x = SpatialCoordinate(mesh)

# Create initial condition
t, dt = 0, 0.01
T = 10*dt
half = Constant(0.5, name="0.5")
u_init = Function(V, name="u^0")
bc_o = DirichletBC(V, Constant(0.0), mf, obstacle)
bc_s = DirichletBC(V, Constant(10), mf, boundary)
a = inner(grad(u0), grad(v))*dx(domain=mesh)
l = Constant(0)*v*dx(domain=mesh)
solve(a == l, u_init, bcs=[bc_o, bc_s])

u_prev = u_init.copy(deepcopy=True)
u_prev.rename("u^(t-1)","")
u_next = u_init.copy(deepcopy=True)
u_next.rename("u^t","")

u_mid = half*u_prev + half*u_next
states = [u_prev.copy(deepcopy=True)]
times  = [float(t)]
timestep = 0
T_file = File("output/T_timedep.pvd")
T_file << u_next

bc = DirichletBC(V, Constant(0), mf, obstacle)

J = 0
while t < T:
    print("Solving for t == %.2e/%.2e" % (t + dt, T))
    f = t/T*x[0]*x[1]
    F = inner((u_next - u_prev)/Constant(dt, name="dt"), v)*dx\
        + inner(grad(u_mid), grad(v))*dx- f*v*dx(domain=mesh)
    solve(F == 0, u_next, J=derivative(F, u_next), annotate=True, bcs=bc)

    T_file << u_next
    u_prev.assign(u_next, annotate=True)
    t += dt

    timestep += 1
    J += dt*assemble(u_next*ds(domain=mesh, subdomain_data=mf,
                               subdomain_id=boundary))
    times.append(float(t))


Jhat = ReducedFunctional(J, Control(s))
Jhat.optimize()
tape.visualise("output/time_dependent_tape.dot", dot=True)
perturbation = project(Expression(("sin(x[0])*x[1]", "cos(x[1])"), degree=2), S)
taylor_test(Jhat, Function(S), perturbation, dJdm=0)
taylor_test(Jhat, Function(S), perturbation)

# perturbation.vector()[:]*=0.01
# plot(mesh)
# ALE.move(mesh, perturbation)
# plot(mesh, color="r")
# plt.show()

# Fix Dirichlet condition
perturbation = project(Expression(("10*x[0]*x[1]*cos(2*pi*x[0])", "8*sin(x[0])*cos(x[1])"),
                                  degree=2), S)
u,v = TrialFunction(S), TestFunction(S)
alpha = 1
a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
bc_b = DirichletBC(S, Constant((0,0)), mf, boundary) 
bc_o = DirichletBC(S, perturbation, mf, obstacle)
representation = Function(S)
solve(lhs(a)==rhs(a), representation, bcs=[bc_b, bc_o])
representation.vector()[:]*=0.01
taylor_test(Jhat, Function(S), representation, dJdm=0)
taylor_test(Jhat, Function(S), representation)


# Fix Neumann condition
perturbation = project(Expression(("10*x[0]*x[1]*cos(2*pi*x[0])", "8*sin(x[0])*cos(x[1])"),
                                  degree=2), S)
u,v = TrialFunction(S), TestFunction(S)
alpha = 1
a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
bc_b = DirichletBC(S, perturbation, mf, boundary) 
bc_o = DirichletBC(S, Constant((0,0)), mf, obstacle)
representation = Function(S)
solve(lhs(a)==rhs(a), representation, bcs=[bc_b, bc_o])
representation.vector()[:]*=0.01
taylor_test(Jhat, Function(S), representation, dJdm=0)
taylor_test(Jhat, Function(S), representation)
