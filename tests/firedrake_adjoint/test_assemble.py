import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake_adjoint import *
from numpy.testing import assert_approx_equal

def test_assemble_0_forms():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V)
    u.assign(4.0)
    a1 = assemble(u * dx)
    a2 = assemble(u**2 * dx)
    a3 = assemble(u**3 * dx)

    # this previously failed when in Firedrake "vectorial" adjoint values
    # where stored as a Function instead of Vector()
    s = a1 + a2 + 2.0 * a3
    rf = ReducedFunctional(s, Control(u))
    # derivative is: (1+2*u+6*u**2)*dx - summing is equivalent to testing with 1
    assert_approx_equal(rf.derivative().vector().sum(), 1. + 2. * 4 + 6 * 16.)

def test_assemble_0_forms_mixed():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V * V)
    u.assign(7.0)
    a1 = assemble(u[0] * dx)
    a2 = assemble(u[0]**2 * dx)
    a3 = assemble(u[0]**3 * dx)

    # this previously failed when in Firedrake "vectorial" adjoint values
    # where stored as a Function instead of Vector()
    s = a1 + 2. * a2 + a3
    s -= a3  # this is done deliberately to end up with an adj_input of 0.0 for the a3 AssembleBlock
    rf = ReducedFunctional(s, Control(u))
    # derivative is: (1+4*u)*dx - summing is equivalent to testing with 1
    assert_approx_equal(rf.derivative().vector().sum(), 1. + 4. * 7)
