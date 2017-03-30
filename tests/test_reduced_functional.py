from fenics import *
from fenics_adjoint import *


def test_constant():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Constant(1)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v))*dx - f**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c**2*u*dx)
    Jhat = ReducedFunctional(J, c)
    assert(taylor_test(Jhat, Constant(5), 1) > 1.9)


def test_function():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    c = Constant(1)
    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    F = inner(grad(u), grad(v))*dx - f**2*v*dx
    solve(F == 0, u, bc)

    J = assemble(c**2*u*dx)
    Jhat = ReducedFunctional(J, f)
    h = Function(V)
    from numpy.random import rand
    h.vector()[:] = rand(V.dim())
    # Note that if you use f.vector() directly, it will not work
    # as expected since f is the control and thus the initial point in control
    # space is changed as you do the test. (Since f.vector is also assigned new values on pertubations)
    g = f.copy(deepcopy=True)
    assert(taylor_test(Jhat, g.vector(), h.vector()) > 1.9)


def test_wrt_function_dirichlet_boundary():
    mesh = UnitSquareMesh(10,10)

    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    class Up(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 1)

    class Down(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0)

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 1)

    left = Left()
    right = Right()
    up = Up()
    down = Down()

    boundary = FacetFunction("size_t", mesh)
    boundary.set_all(0)
    up.mark(boundary, 1)
    down.mark(boundary,2)
    ds = Measure("ds", subdomain_data=boundary)

    bc_func = project(Expression("sin(x[1])", degree=1), V)
    bc1 = DirichletBC(V,bc_func,left)
    bc2 = DirichletBC(V,2,right)
    bc = [bc1,bc2]

    g1 = Constant(2)
    g2 = Constant(1)
    f = Function(V)
    f.vector()[:] = 10

    a = inner(grad(u), grad(v))*dx
    L = inner(f,v)*dx + inner(g1,v)*ds(1) + inner(g2,v)*ds(2)

    solve(a==L,u_,bc)

    J = assemble(u_**2*dx)

    Jhat = ReducedFunctional(J, bc1)
    h = Function(V)
    h.vector()[:] = 1
    assert(taylor_test(Jhat, bc_func, h, dot=lambda a, b: a.vector().inner(b)) > 1.9)

