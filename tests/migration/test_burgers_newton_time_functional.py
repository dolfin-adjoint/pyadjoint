import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def test_main():
    n = 30
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh, "CG", 2)

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    m = Control(ic)

    def Dt(u, u_, timestep):
        return (u - u_)/timestep

    u_ = ic.copy(deepcopy=True)
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    j = 0
    j += 0.5*AdjFloat(timestep)*assemble(u_*u_*u_*u_*dx)

    while (t <= end):
        solve(F == 0, u, bc)
        u_.assign(u)

        t += float(timestep)

        if t>end:
            quad_weight = AdjFloat(0.5)
        else:
            quad_weight = AdjFloat(1.0)
        j += quad_weight*AdjFloat(timestep)*assemble(u_*u_*u_*u_*dx)

    h = Function(V)
    h.vector()[:] = rand(V.dim())*1.4
    dJdm = compute_gradient(j, m)._ad_dot(h)
    HJm  = Hessian(j, m)
    Hm = HJm(h)._ad_dot(h)

    minconv = taylor_test(ReducedFunctional(j, m), ic, h, dJdm=dJdm, Hm=Hm)
    assert minconv > 2.9