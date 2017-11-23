"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

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

def main(ic, annotate=False):

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
        solver.solve(burgers, u.vector(), annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

    return u_

if __name__ == "__main__":

    ic = project(Expression("sin(2*pi*x[0])", degree=1),  V)
    forward = main(ic, annotate=True)

    J = assemble(forward*forward*dx)
    m = Control(ic)
    Jhat = ReducedFunctional(J, m)
    dJdm = compute_gradient(J, m)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJdm = h._ad_dot(dJdm)

    def Jfunc(ic):
        forward = main(ic, annotate=False)
        return assemble(forward*forward*dx)

    minconv = taylor_test(Jfunc, ic, h, dJdm)
    assert minconv > 1.9
