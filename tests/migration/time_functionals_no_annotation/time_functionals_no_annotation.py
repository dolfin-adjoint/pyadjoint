from fenics import *
from fenics_adjoint import *

mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "R", 0)
f = Constant(1.0)

# Solve the 'PDE' du/dt = 1
# with initial condition u(0) = c
# this gives the solution u(t) = c + t.
def run_forward(annotate=True):
    u = TrialFunction(V)
    v = TestFunction(V)

    u_0 = Function(V, name="Value")
    u_0.vector()[0] = 1.0

    dt = 0.5
    T =  1.0

    F = ( (u - u_0)/dt*v - f*v)*dx
    a, L = lhs(F), rhs(F)

    t = float(dt)

    while t <= T:

        solve(a == L, u_0, annotate=annotate)

        t += float(dt)

    return u_0

if __name__ == "__main__":

    u = run_forward()

    # Pointwise evaluation (at the end of time, symbolically)
    J = assemble(inner(u,u)*dx)
    dJdic = compute_gradient(J, Control(u))
    # Work out the solution by hand -- it's 4
    assert dJdic.vector().array()[0] == 4.0
