import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake_adjoint import *

from numpy.random import rand
from numpy.testing import assert_approx_equal, assert_allclose

def test_constant_assign_function_assign():
    af = AdjFloat(0.1)
    cst = Constant(0.0)
    cst.assign(af)
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.assign(cst)
    J = assemble(u*u*dx)
    rf = ReducedFunctional(J, Control(af))
    taylor_test(rf, AdjFloat(1.0), AdjFloat(0.01))

def test_constant_assign_function_project():
    af = AdjFloat(0.1)
    cst = Constant(0.0)
    cst.assign(af)
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.project(cst)
    J = assemble(u*u*dx)
    rf = ReducedFunctional(J, Control(af))
    taylor_test(rf, AdjFloat(1.0), AdjFloat(0.01))

def test_constant_assign_assemble():
    af = AdjFloat(0.1)
    cst = Constant(0.0)
    cst.assign(af)
    mesh = UnitSquareMesh(1, 1)
    J = assemble(cst*cst*dx(domain=mesh))
    rf = ReducedFunctional(J, Control(af))
    taylor_test(rf, AdjFloat(1.0), AdjFloat(0.01))

