import pytest

pytest.importorskip("fenics")
pytest.importorskip("fenics_adjoint")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand

fexp = Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)*sin(t)", t=0, degree=4)
mesh = UnitSquareMesh(4, 4)
V = FunctionSpace(mesh, "CG", 1)

dt = Constant(0.1)
T = 1.0

def main():
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)

    # Create a list with all control functions
    ctrls = {}
    t = float(dt)
    while t <= T:
        fexp.t = t
        ctrls[t] = project(fexp, V, annotate=True)
        t += float(dt)

    u_0 = Function(V, name="Solution")
    u_1 = Function(V, name="NextSolution")

    F = ( (u - u_0)/dt*v + inner(grad(u), grad(v)) + f*v)*dx
    a, L = lhs(F), rhs(F)
    bc = DirichletBC(V, 1.0, "on_boundary")

    t = float(dt)
    J = 0.5*AdjFloat(dt)*assemble(u_0**2*dx)
    while t <= T:
        f.assign(ctrls[t], annotate=True)
        solve(a == L, u_0, bc)
        t += float(dt)

        if t > T:
            quad_weight = 0.5
        else:
            quad_weight = 1.0

        J += quad_weight*AdjFloat(dt)*assemble(u_0**2*dx)

    return J, list(ctrls.values())

def test_heat():
    J, ctrls = main()

    regularisation = sum([(new-old)**2 for new, old in zip(ctrls[1:], ctrls[:-1])])
    regularisation = regularisation*dx

    alpha = Constant(1e0)
    J = J + assemble(alpha*regularisation)
    m = [Control(c) for c in ctrls]

    hs = []
    for _ in ctrls:
        h = Function(V)
        h.vector()[:] = rand(V.dim())
        hs.append(h)

    rf = ReducedFunctional(J, m)
    minconv = taylor_test(rf, ctrls, hs)

    assert minconv > 1.9

if __name__ == "__main__":
    test_heat()
