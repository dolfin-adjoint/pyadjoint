from firedrake import *
from firedrake_adjoint import *
from numpy.testing import assert_approx_equal
from taylor_test import taylor_adjoint

mesh = UnitSquareMesh(5, 5)
V1 = FunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)

X = V1 * V2

u = Function(X)
v = TestFunction(X)
v1 = TestFunction(V1)

f = Function(X)
f.sub(0).assign(as_ufl(1))
g = Function(V1).assign(f.sub(0))

def J(f):
    F = (-inner(u,v) + inner(grad(u), grad(v)))*dx - inner(f,v)*dx
    solve(F == 0, u)
    u1, u2 = u.split()

    # Working if we don't modify one of the split functions (i.e. if we comment the 2 next lines)
    F = (-inner(u1,v1) + inner(grad(u1), grad(v1)))*dx - inner(g,v1)*dx
    solve(F == 0, u1)
    return assemble(u1**2*dx)

J0 = J(f)
rf = ReducedFunctional(J0, Control(f))
assert_approx_equal(rf(f), J0)

taylor_adjoint(J, f)
