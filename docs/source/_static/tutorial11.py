from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *

n = 30
mesh = UnitSquareMesh(n, n)
V = VectorFunctionSpace(mesh, "CG", 2)

ic = project(Expression(("sin(2*pi*x[0])", "cos(2*pi*x[1])"), degree=2),  V)
nu = Constant(0.0001, name="nu")

def main(ic):
    u = ic.copy(deepcopy=True, name="Velocity")
    u_next = Function(V, name="VelocityNext")
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
    u = main(ic)

    gst = compute_gst("Velocity", "Velocity", nsv=3)
    for i in range(gst.ncv):
        sigma = gst.get_gst(i)
        print("Growth rate of vector %s: %s" % (i, sigma))
