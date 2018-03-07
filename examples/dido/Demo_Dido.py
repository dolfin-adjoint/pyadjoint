from dolfin import *
from dolfin_adjoint import *
from femorph import *
import matplotlib.pyplot as plt

N = 100
mesh = Mesh(UnitSquareMesh(N, N))
One = Constant(1.0)

V = VectorFunctionSpace(mesh, "CG", 1)
s0 = Function(V)
ALE.move(mesh, s0)
print(assemble(One*dx(domain=mesh)))

alpha = 1e1
J = -assemble(One*dx(domain=mesh)) + alpha*(assemble(One*ds(domain=mesh)) - 4)**2
Jhat = ReducedFunctional(J, Control(s0))

s = interpolate(Expression(("10*x[0]", "10*x[1]"), degree=2), V)
taylor_test(Jhat, s0, s, dJdm=0)
taylor_test(Jhat, s0, s)


it, max_it = 1, 100
min_stp = 1e-6
red_tol = 1e-6
red = 2*red_tol
move = Function(V)
move_step = Function(V)
Js = [Jhat(move)]
meshout = File("output/mesh.pvd")
meshout << mesh

def riesz_representation(value):
    V = VectorFunctionSpace(mesh, "CG", 1)
    u,v = TrialFunction(V), TestFunction(V)
    alpha = 1
    a = Constant(alpha)*inner(grad(u), grad(v))*dx+inner(u,v)*dx
    A = assemble(a)
    representation = Function(V)
    solve(A, representation.vector(), value)
    return representation


while it <= max_it and red > red_tol:
    print("-"*5, "Iteration %d" %it, "-"*5)

    # Compute derivative of previous configuration
    dJdm = Jhat.derivative(options={"riesz_representation":
                                    riesz_representation})

    # Linesearch
    step = 5
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
