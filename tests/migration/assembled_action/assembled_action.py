from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 1)

def main(data):
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = inner(u, v)*dx
    M = assemble(mass)

    rhs = M*data.vector()
    soln = Function(V)

    solve(M, soln.vector(), rhs)
    return soln

if __name__ == "__main__":
    data = Function(V, name="Data")
    data.vector()[0] = 1.0

    soln = main(data)

    J = assemble(inner(soln, soln)*dx)
    c = Control(data)
    dJdic = compute_gradient(J, c)

    def Jhat(data):
        soln = main(data)
        return assemble(soln*soln*dx)

    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJdic = h._ad_dot(dJdic)

    minconv = taylor_test(Jhat, data, h, dJdic)
    assert minconv > 1.9
