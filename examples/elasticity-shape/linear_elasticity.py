from dolfin import *
import matplotlib.pyplot as plt
from dolfin_adjoint import *
from femorph import *
n = 50
length = 1
x0 = 0.0
x1 = x0 + length
height = 0.2
y0 = 0.0
y1 = y0 + height

mesh_1 = RectangleMesh(Point(x0, y0), Point(x1, y1), int((x1-x0)*n), int((y1-y0)*n), 'crossed')
mesh = Mesh(mesh_1)


boundary_parts = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
left  = AutoSubDomain(lambda x: near(x[0], x0))
right = AutoSubDomain(lambda x: near(x[0], x1))
left .mark(boundary_parts, 1)
right.mark(boundary_parts, 2)

# Rotation rate and mass density
rho = 1.0
g = 9.81
# Loading due to gravity
f = Expression(("0", "-rho*g"),
               g=g, rho=rho, degree=2)

# Elasticity parameters
E = 5e2
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)
s = Function(V)
ALE.move(mesh, s)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), grad(v))*dx
L = inner(f, v)*dx

# Set up boundary condition on inner surface
c = Constant((0.0, 0.0))
bcLeft = DirichletBC(V, c, boundary_parts, 1)
bcRight = DirichletBC(V, c, boundary_parts, 2)
# Create solution function
u_fin = Function(V, name="deform")
solve(a==L, u_fin, bcs=[bcLeft])
# solve(a==L, u_fin, bcs=[bcLeft, bcRight])
elas = File("elasticity.pvd", "compressed")
elas << u_fin

# Project and write stress field to post-processing file
# W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
# import dolfin
# stress = dolfin.project(sigma(u), V=W)
# File("stress.pvd") << stress
alpha_reg = 1e1
#alpha_vol = 1e1
#alpha_bary = 1e1
#one = interpolate(Constant(1.0), FunctionSpace(mesh, "CG", 1))
x = SpatialCoordinate(mesh)
J = assemble(inner(u_fin,u_fin)*dx)
#J += alpha_bary*(assemble(x[0]*dx)/assemble(one*dx)-length/2)*(assemble(x[0]*dx)/assemble(one*dx)-length/2)
J += alpha_reg*assemble((x1-x0-x[0])**2*inner(s,s)*ds)
#J += alpha_vol*(assemble(one*dx)-height*length)*(assemble(one*dx)-height*length)
Jhat = ReducedFunctional(J, Control(s))
Jhat.optimize()

n = VolumeNormal(mesh)
s2 = Function(V)
s2.interpolate(Expression(("0","0.02*x[0]*x[0]*(length-x[0])*(length-x[0])"), length=length, degree=2))
# [bc.apply(s2.vector()) for bc in [bcLeft, bcRight]]
s2.interpolate(Expression(("5*x[0]","5*x[0]*x[1]"), x0=x0, degree=2))
[bc.apply(s2.vector()) for bc in [bcLeft]]
s0 = Function(V)
taylor_test(Jhat, s0, s2, dJdm=0)
print("-"*10)
taylor_test(Jhat, s0, s2)

def riesz_representation(value):
    u,v = TrialFunction(V), TestFunction(V)
    alpha = 1
    a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
    A = assemble(a)
    # [bc.apply(A,value) for bc in [bcLeft, bcRight]]
    [bc.apply(A,value) for bc in [bcLeft]]
    #print("Barycenter: %.2e" %(assemble(x[0]*dx)/assemble(one*dx)))
    representation = Function(V)
    solve(A, representation.vector(), value)
    return representation


it, max_it = 1, 100
min_stp = 1e-6
red_tol = 1e-3
red = 2*red_tol
meshout = File("output/mesh.pvd")

move = Function(V)
move_step = Function(V)
Js = [Jhat(move)]
meshout << mesh

while it <= max_it and red > red_tol:
    print("-"*5, "Iteration %d" %it, "-"*5)

    # Compute derivative of previous configuration
    dJdm = Jhat.derivative(options={"riesz_representation":
                                    riesz_representation})

    # Linesearch
    step = 1e-1
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

print("Relative reduction: %.2e" %red)
print("-"*5, "Optimization Finished", "-"*5)
print(Js[0], Js[-1])
print(Jhat(s),Jhat(move))
solve(a==L, u_fin, bcs=[bcLeft])
print(assemble(inner(u_fin,u_fin)*dx)+alpha_reg*assemble((x1-x0-x[0])**2*inner(move,move)*ds))

elas << u_fin

