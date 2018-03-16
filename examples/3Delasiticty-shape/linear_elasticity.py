from dolfin import *
import matplotlib.pyplot as plt
from dolfin_adjoint import *
from femorph import *
n = 5
length = 1 #m
width = 0.2 #m 
height = 0.05 #m
x0 = 0.0
y0 = 0.0
z0 = 0.0
x1 = x0 + length
z1 = z0 + height
y1 = y0 + width

def geometry_3d():
    """Prepares 3D geometry. Returns facet function with 1, 2 on parts of
    the boundary."""
    mesh = Mesh(BoxMesh(Point(x0,y0,z0),Point(x1,y1,z1), int(length/height*n),
                        int(width/height*n),n))
    gdim = mesh.geometry().dim()
    X0 = mesh.coordinates()[:, 0].min()
    X1 = mesh.coordinates()[:, 0].max()
    Y0 = mesh.coordinates()[:, 1].min()
    Y1 = mesh.coordinates()[:, 1].max()
    Z0 = mesh.coordinates()[:, 2].min()
    Z1 = mesh.coordinates()[:, 2].max()
    boundary_parts = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
    left  = AutoSubDomain(lambda x: near(x[0], X0))
    right = AutoSubDomain(lambda x: near(x[0], X1))
    top = AutoSubDomain(lambda x: near(x[2], Z1))
    top.mark(boundary_parts, 3)
    left.mark(boundary_parts, 1)
    right.mark(boundary_parts, 2)
    return mesh, boundary_parts


mesh, boundary_parts = geometry_3d()


# Loading due to gravity
rho = 2700 #kg/m^3 aluminium
g = 9.81
f = Expression(("0", "0", "-rho*g"),
               g=g, rho=rho, degree=2)

# Elasticity parameters
E = 6.894757e10 # Youngs modulus aluminium
nu = 0.33 # Poisson ratio aluminium
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)
s = Function(V)
ALE.move(mesh, s)

stress = Expression(("0", "0", "-4e7*exp(-pow((x[0]-length/5),2)/0.01)*exp(-pow((x[1]-width/2),2)/0.001)"), width=width, length=length, degree=2, domain=mesh) # N/m^2s
dS_stress = Measure("ds", domain=mesh, subdomain_data=boundary_parts)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v))*dx
L = inner(f, v)*dx  + inner(stress, v)*dS_stress(3)

# Set up boundary condition on inner surface
c = Constant((0.0, 0.0, 0.0))
bcLeft = DirichletBC(V, c, boundary_parts, 1)
bcRight = DirichletBC(V, c, boundary_parts, 2)

# Create solution function
u_fin = Function(V, name="deform")
solve(a==L, u_fin, bcs=[bcLeft, bcRight], solver_parameters={"linear_solver": "mumps"})

elas = File("elasticity_new_functional.pvd", "compressed")
elas << u_fin
J = assemble(inner(sigma(u_fin),sym(grad(u_fin)))*dx)
Jhat = ReducedFunctional(J, Control(s))
Jhat.optimize()
# s2 = Function(V)
# s2.interpolate(Expression(("0", "0", "sin(2*pi/length*x[0])"), degree=2, length=length))
# [bc.apply(s2.vector()) for bc in [bcLeft, bcRight]]

# s0 = Function(V)
# s2.vector()[:]*=0.01
# ALE.move(mesh,s2)
# plot(mesh)
# plt.show()
# taylor_test(Jhat, s0, s2, dJdm=0)
# print("-"*10)
# taylor_test(Jhat, s0, s2)
# exit(1)

tape.visualise("tape_linear.dot", dot=True)

def riesz_representation(value):
    u,v = TrialFunction(V), TestFunction(V)
    alpha = 1
    a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
    A = assemble(a)
    [bc.apply(A,value) for bc in [bcLeft, bcRight]]
    representation = Function(V)
    solve(A, representation.vector(), value)
    return representation

it, max_it = 1, 15
red_tol = 1e-2
red = 2*red_tol
meshout = File("output/mesh_new.pvd")

move = Function(V)
move_step = Function(V)
Js = [Jhat(move)]
meshout << mesh
stress_out = Function(V)
bc_stress = DirichletBC(V, stress, boundary_parts, 3)
bc_stress.apply(stress_out.vector())
stress_file = File("stress.pvd")
stress_file << stress_out
while it <= max_it and red > red_tol:
    print("-"*5, "Iteration %d" %it, "-"*5)

    # Compute derivative of previous configuration
    dJdm = Jhat.derivative(options={"riesz_representation":
                                    riesz_representation})

    # Linesearch
    step = 1e-7
    min_stp = step/(2**16)
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
    print("Relative reduction: %.2e" %red)
    print("Functional value: %.2e" %Js[-1])
    print("offset: %.2e" %(  (assemble(1*dx(domain=mesh))/(length*height*width))))


    it += 1

print("Relative reduction: %.2e" %red)
print("-"*5, "Optimization Finished", "-"*5)
print(Js[0], Js[-1])
print(Jhat(s0),Jhat(move))

solve(a==L, u_fin, bcs=[bcLeft, bcRight])
stress_file << stress_out
elas << u_fin
