from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def main(m):
    a = interpolate(Constant(1), m.function_space())
    z = Function(m.function_space(), name="z")
    z.assign(0.5 * a + 2.0 * m)

    return z

if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    m = interpolate(Expression("x[0]", degree=1), V)
    z = main(m)

    c = Control(m)
    J = assemble(inner(z, z)*dx)

    dJ = compute_gradient(J, c)
    h = Function(V)
    h.vector()[:] = rand(V.dim())
    dJ = h._ad_dot(dJ)

    def Jhat(m):
        z = main(m)
        return assemble(inner(z, z)*dx)

    minconv = taylor_test(Jhat, c, h, dJ)
