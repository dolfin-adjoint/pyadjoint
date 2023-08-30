import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake.adjoint import *
import numpy as np

mesh = UnitSquareMesh(2, 2)
cg2 = FiniteElement("CG", triangle, 2)
cg1 = FiniteElement("CG", triangle, 1)
ele = MixedElement([cg2, cg1])
ZZ = FunctionSpace(mesh, ele)
V2 = FunctionSpace(mesh, cg2)


# the tests are run on functions from the MixedFunctionSpace ZZ
# and on a normal (non-mixed) FunctionSpace. Calling split() on
# a non-mixed function is trivial, but was previously broken
@pytest.fixture(params=[ZZ, V2], ids=('mixed', 'non-mixed'))
def Z(request):
    return request.param


rg = RandomGenerator()


def main(ic, fnsplit=True):
    u = Function(V2)
    w = TrialFunction(V2)
    v = TestFunction(V2)

    if fnsplit:
        ic_u = ic.split()[0]
    else:
        ic_u = split(ic)[0]

    mass = inner(w, v)*dx
    rhs  = inner(ic_u, v)*dx

    solve(mass == rhs, u)

    return u


def test_split(Z):
    ic = Function(Z)

    u = main(ic, fnsplit=False)
    j = assemble(u**2*dx)
    rf = ReducedFunctional(j, Control(ic))

    h0 = Function(Z).assign(1.)
    assert taylor_test(rf, ic.copy(deepcopy=True), h=h0) > 1.9


def test_fn_split(Z):
    set_working_tape(Tape())
    ic = Function(Z)

    u = main(ic, fnsplit=True)
    j = assemble(u**2*dx)
    rf = ReducedFunctional(j, Control(ic))

    h = rg.uniform(Z)

    assert taylor_test(rf, ic, h) > 1.9


def test_fn_split_hessian(Z):
    set_working_tape(Tape())
    ic = Function(Z)

    u = main(ic, fnsplit=True)
    j = assemble(u ** 4 * dx)
    rf = ReducedFunctional(j, Control(ic))

    h = rg.uniform(Z)
    dJdm = rf.derivative()._ad_dot(h)
    Hm = rf.hessian(h)._ad_dot(h)
    assert taylor_test(rf, ic, h, dJdm=dJdm, Hm=Hm) > 2.9


def test_fn_split_no_annotate(Z):
    set_working_tape(Tape())
    ic = Function(Z)

    u = Function(V2)
    w = TrialFunction(V2)
    v = TestFunction(V2)

    ic_u = ic.split(annotate=True)[0]
    ic_uv = ic.split(annotate=False)[0]

    mass = inner(w, v) * dx
    rhs = inner(ic_u, v) * dx

    solve(mass == rhs, u)
    j = assemble(u ** 4 * dx + ic_uv * dx)
    rf = ReducedFunctional(j, Control(ic))

    h = rg.uniform(Z)
    r = taylor_to_dict(rf, ic, h)

    assert min(r["R0"]["Rate"]) > 0.9
    assert min(r["R1"]["Rate"]) > 1.9
    assert min(r["R2"]["Rate"]) > 2.9


def test_split_subvariables_update(Z):
    z = Function(Z)
    u = z.split()[0]
    u.project(Constant(1.))
    assert np.allclose(z.sub(0).vector().dat.data, u.vector().dat.data)

def test_merge_blocks():
    mesh = UnitSquareMesh(1,1)
    V = FunctionSpace(mesh, 'CG', 1)
    W = V * V
    w = Function(W)
    w1, w2 = w.split()
    w1_const = Constant(0.1, domain=mesh)
    w2_const = Constant(0.2, domain=mesh)
    w1.project(w1_const)
    w2.project(w2_const)
    J = assemble(w1*w1*dx)
    c = Control(w1_const)
    rf = ReducedFunctional(J, c)
    assert taylor_test(rf, Constant(0.3, domain=mesh), Constant(0.01, domain=mesh)) > 1.95


