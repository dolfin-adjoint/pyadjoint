from fenics import *
from fenics_adjoint import *
import os
import os.path

if False:
    # TODO: Consider if something similar to this needs implementing in pyadjoint.
    dolfin.parameters["adjoint"]["cache_factorizations"] = True
    if dolfin.__version__ > '1.2.0':
        dolfin.parameters["adjoint"]["symmetric_bcs"] = True

n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, annotate=False):

    u_ = ic.copy(deepcopy=True)
    u = Function(V, name="VelocityNext")
    v = TestFunction(V)

    nu = Constant(0.0001, name="nu")

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = 0.0
    end = 0.2
    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

    return u_, nu

if __name__ == "__main__":
    cache_file = "cache.pck"

    try:
        os.remove(cache_file)
    except OSError:
        pass

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    forward, nu = main(ic, annotate=True)

    J = assemble(forward*forward*dx + ic*ic*dx)

    m1 = Control(ic)
    m2 = Control(nu)

    # Test caching of the reduced functional evaluation
    rf = ReducedFunctional(J, [m1, m2], cache=cache_file)

    t = dolfin.Timer("")
    a = rf([interpolate(Constant(2), V), Constant(4)])
    time_a = t.stop()

    t = dolfin.Timer("")
    b = rf([interpolate(Constant(2), V), Constant(4)])
    time_b = t.stop()

    assert a == b
    assert time_a/time_b > 50 # Check that speed-up is significant

    # Now let's test the caching of the functional gradient
    t = dolfin.Timer("")
    a = rf.derivative()
    time_a = t.stop()

    t = dolfin.Timer("")
    b = rf.derivative()
    time_b = t.stop()

    assert max(abs(a[0].vector().array() - b[0].vector().array())) == 0
    assert float(a[1]) == float(b[1])
    assert time_a/time_b > 100 # Check that speed-up is significant

    # Finally, let's check caching of the Hessian
    # TODO

    # Check that the cache file gets created
    del rf
    assert os.path.isfile(cache_file)
