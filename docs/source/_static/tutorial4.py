from dolfin import *
from fenics_adjoint import *

n = 30
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)

ic = project(Expression(("sin(2*pi*x[0])", "cos(2*pi*x[1])"), degree=2),  V)

def main(nu):
    u = ic.copy(deepcopy=True)
    u_next = Function(V)
    v = TestFunction(V)

    timestep = Constant(0.01)

    F = (inner((u_next - u)/timestep, v)
       + inner(grad(u_next)*u_next, v)
       + nu*inner(grad(u_next), grad(v)))*dx

    bc = DirichletBC(V, (0.0, 0.0), "on_boundary")

    t = 0.0
    end = 0.1
    while (t <= end):
        solve(F == 0, u_next, bc)
        u.assign(u_next)
        t += float(timestep)

    return u

if __name__ == "__main__":
    nu = Constant(0.0001)
    u = main(nu)

    J = assemble(inner(u, u)*dx)
    dJdnu = compute_gradient(J, nu)
    
    h_nu = Constant(1) #the direction of the perturbation
    Jhat_nu = ReducedFunctional(J,nu) #the functional as a pure function of nu
    conv_rate_nu = taylor_test(Jhat_nu, Constant(nu), h_nu,dJdm = float(dJdnu))

    # h_u = Function(V)
    # h_u.vector()[:] = 0.1
    # Jhat_u = ReducedFunctional(J,u)
    # conv_rate_u = taylor_test(Jhat_u,u.copy(deepcopy=True),h_u)
