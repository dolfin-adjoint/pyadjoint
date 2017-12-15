from fenics import *
from fenics_adjoint import *

mesh = UnitIntervalMesh(30)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_) / timestep

pr = project(Expression("sin(2*pi*x[0])", degree=1), V)
u_ = Function(V)
u_.vector()[:] = pr.vector()[:]
u = Function(V)
v = TestFunction(V)
nu = Constant(0.0001)
a = Constant(0.4)
timestep = AdjFloat(0.05)
bc = DirichletBC(V, 0.0, "on_boundary")
t = 0.0

F = (Dt(u, u_, timestep) * v
     + a * u * u.dx(0) * v + nu * u.dx(0) * v.dx(0)) * dx

end = 0.3
J = 0
while (t <= end):
    solve(F == 0, u, bc)
    u_.assign(u)
    t += float(timestep)
    J += timestep*assemble(u_*u_*dx)

h = Function(V)
h.vector()[:] = 1
Jhat = ReducedFunctional(J, u_)
print taylor_test(Jhat, pr.copy(deepcopy=True), h)
