"""Verify the tangent linear model of example 5.2 in 10.1007/s10589-009-9307-9"""

from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *

n = 512
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
gamma = Constant(1e3) # 1.0 / alpha in the paper

f = interpolate(Expression("-std::abs(x[0]*x[1] - 0.5) + 0.25"), V)
yd = Function(f)
alpha = 1e-2
u = Function(V, name="Control")

def main():
    y = Function(V, name="Solution")
    w = TestFunction(V)

    eps = 1e-4

    def smoothmax(r):
        return conditional(gt(r, eps), r - eps/2, conditional(lt(r, 0), 0, r**2 / (2*eps)))
    def uflmax(a, b):
        return conditional(gt(a, b), a, b)

    F = inner(grad(y), grad(w))*dx - gamma * inner(smoothmax(-y), w)*dx - inner(f + u, w)*dx

    bc = DirichletBC(V, 0.0, "on_boundary")

    solve(F == 0, y, bcs = bc, solver_parameters={"newton_solver": {"maximum_iterations": 30}})
    return y

if __name__ == "__main__":
    fwd = Timer("forward")
    y = main()
    fwd_time = fwd.stop()

    print("fwd_time: ", fwd_time)

    J = Functional(0.5*inner(y - yd, y - yd)*dx*dt[FINISH_TIME] + alpha/2*inner(u, u)*dx*dt[START_TIME])
    m = TimeConstantParameter(u)

    adj = Timer("adjoint")
    dJ = compute_gradient(J, m)
    adj_time = adj.stop()

    print("adj_time: ", adj_time)
    print("ratio: ", (fwd_time + adj_time) / (fwd_time))
