from fenics import *
from fenics_adjoint import *

from numpy.random import rand


n = 30
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)


def Dt(u, u_, timestep):
    return (u - u_)/timestep


class BurgersProblem(NonlinearProblem):
    def __init__(self, F, u, bc):
        NonlinearProblem.__init__(self)
        self.f = F
        self.jacob = derivative(F, u)
        self.bc = bc

    def F(self, b, x):
        assemble(self.f, tensor=b)
        self.bc.apply(b)

    def J(self, A, x):
        assemble(self.jacob, tensor=A)
        self.bc.apply(A)


def test_burgers_oo():
    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)

    u_ = ic.copy(deepcopy=True)
    u = Function(V)
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    solver = NewtonSolver()
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6

    bc = DirichletBC(V, 0.0, "on_boundary")
    burgers = BurgersProblem((Dt(u, u_, timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx, u, bc)

    t = 0.0
    end = 0.2
    while (t <= end):
        solver.solve(burgers, u.vector())
        u_.assign(u)

        t += float(timestep)

    forward = u_

    J = assemble(forward*forward*dx)
    m = Control(ic)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJdm = h._ad_dot(compute_gradient(J, m))
    Jhat = ReducedFunctional(J, m)
    minconv = taylor_test(Jhat, ic, h, dJdm=dJdm)

    assert minconv > 1.9