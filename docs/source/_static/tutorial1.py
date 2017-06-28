from dolfin import *

n = 30
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)

ic = project(Expression(("sin(2*pi*x[0])", "cos(2*pi*x[1])"), degree=2),  V)
u = ic.copy(deepcopy=True)
u_next = Function(V)
v = TestFunction(V)

nu = Constant(0.0001)

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
