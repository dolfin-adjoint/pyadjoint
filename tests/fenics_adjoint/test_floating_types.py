import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand


def test_loop_split():
    mesh = UnitSquareMesh(10, 10)
    V_element = VectorElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, V_element)

    f = Function(V)
    u_, p = f.split()
    c = Constant([1, 1])
    control = Control(c)
    bc = DirichletBC(V, Constant([1, 1]), "on_boundary")

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx

    J = 0
    for i in range(3):
        L = Constant(i+1)*inner(c, v)*dx
        solve(a == L, f, bc)
        J += assemble(u_**4*dx)

    Jhat = ReducedFunctional(J, control)

    h = Constant([1, 1])

    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))

    assert taylor_test(Jhat, c, h, dJdm=dJdm) > 1.9
    assert taylor_test(Jhat, c, h, dJdm=dJdm, Hm=Hm) > 2.9


def test_split_control():
    mesh = UnitSquareMesh(10, 10)
    V_element = VectorElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, V_element)

    f = Function(V)
    u_, p = f.split()
    c = Constant([1, 1])
    bc = DirichletBC(V, Constant([1, 1]), "on_boundary")

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx

    J = 0
    for i in range(3):
        L = Constant(i)*inner(c, v)*dx
        solve(a == L, f, bc)
        if i == 0:
            control = Control(f)
            c_value = f.copy(deepcopy=True)
        J += assemble(u_**4*dx)

    Jhat = ReducedFunctional(J, control)

    h = Function(V)
    h.vector()[:] = rand(V.dim())

    dJdm = h._ad_dot(Jhat.derivative())
    Hm = h._ad_dot(Jhat.hessian(h))

    assert taylor_test(Jhat, c_value, h, dJdm=dJdm) > 1.9
    assert taylor_test(Jhat, c_value, h, dJdm=dJdm, Hm=Hm) > 2.9
