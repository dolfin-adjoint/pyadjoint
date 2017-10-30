from fenics import *
from fenics_adjoint import *


if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    a = Constant(0.5)
    b = Constant(0.25)

    f = Expression("(a-x[0])*(a-x[0])*b*b", a=a, b=b, degree=2)
    f.dependencies = [a, b]

    dfda = Expression("2*(a-x[0])*b", a=a, b=b, degree=1)
    dfdb = Expression("2*b*(a-x[0])*(a-x[0])", a=a, b=b, degree=2)

    f.user_defined_derivatives = {a: dfda, b: dfdb}

    J = assemble(f**2*dx(domain=mesh))
    rf1 = ReducedFunctional(J, Control(a))
    rf2 = ReducedFunctional(J, Control(b))

    h = Constant(1.0)
    assert taylor_test(rf1, a, h) > 1.9
    assert taylor_test(rf2, b, h) > 1.9

    rf3 = ReducedFunctional(J, [Control(a), Control(b)])
    hs = [Constant(1.0), Constant(1.0)]
    assert taylor_test_multiple(rf3, [a, b], hs) > 1.9
