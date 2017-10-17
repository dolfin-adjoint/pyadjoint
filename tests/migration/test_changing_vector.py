import pytest
from fenics import *
from fenics_adjoint import *

from numpy import array


n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

@pytest.mark.xfail(reason='How pyadjoint should behave in this problem has currently not been decided.')
def test_changing_vector():
    ic = project(Expression("sin(2*pi*x[0])", degree=1), V)
    u_ = ic.copy(deepcopy=True, name="Velocity")
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Function(V)
    nu.vector()[:] = 0.0001

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    while (t <= end):
        nu.vector()[:] += array([0.0001] * V.dim()) # <--------- change nu here by hand
        solve(F == 0, u, bc)
        u_.assign(u)

        t += float(timestep)

    print("Running forward replay .... ")
    success = replay_dolfin(forget=False) # <------ should catch it here
    assert not success