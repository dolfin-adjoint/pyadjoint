import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

from numpy.random import rand

mesh = UnitSquareMesh(2, 2)
cg2 = FiniteElement("CG", triangle, 2)
cg1 = FiniteElement("CG", triangle, 1)
ele = MixedElement([cg2, cg1])
Z = FunctionSpace(mesh, ele)
V2 = FunctionSpace(mesh, cg2)

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


def test_split():
    ic = Function(Z)

    u = main(ic, fnsplit=False)
    j = assemble(u**2*dx)
    rf = ReducedFunctional(j, Control(ic))

    assert taylor_test(rf, ic.copy(deepcopy=True), h=project(Constant([1, 1]), ic.function_space())) > 1.9


def test_fn_split():
    set_working_tape(Tape())
    ic = Function(Z)

    u = main(ic, fnsplit=True)
    j = assemble(u**2*dx)
    rf = ReducedFunctional(j, Control(ic))

    h = Function(Z)
    h.vector()[:] = rand(Z.dim())
    assert taylor_test(rf, ic, h) > 1.9


def test_fn_split_hessian():
    set_working_tape(Tape())
    ic = Function(Z)

    u = main(ic, fnsplit=True)
    j = assemble(u ** 4 * dx)
    rf = ReducedFunctional(j, Control(ic))

    h = Function(Z)
    h.vector()[:] = rand(Z.dim())
    dJdm = rf.derivative()._ad_dot(h)
    Hm = rf.hessian(h)._ad_dot(h)
    assert taylor_test(rf, ic, h, dJdm=dJdm, Hm=Hm) > 2.9


def test_fn_split_deepcopy():
    set_working_tape(Tape())
    ic = Function(Z)

    u = Function(V2)
    w = TrialFunction(V2)
    v = TestFunction(V2)

    ic_u = ic.split(deepcopy=True)[0]

    mass = inner(w, v) * dx
    rhs = inner(ic_u, v) * dx

    solve(mass == rhs, u)
    j = assemble(u ** 4 * dx)
    rf = ReducedFunctional(j, Control(ic))

    h = Function(Z)
    h.vector()[:] = rand(Z.dim())
    r = taylor_to_dict(rf, ic, h)

    assert min(r["FD"]["Rate"]) > 0.9
    assert min(r["dJdm"]["Rate"]) > 1.9
    assert min(r["Hm"]["Rate"]) > 2.9


def test_fn_split_no_annotate():
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

    h = Function(Z)
    h.vector()[:] = rand(h.function_space().dim())
    r = taylor_to_dict(rf, ic, h)

    assert min(r["FD"]["Rate"]) > 0.9
    assert min(r["dJdm"]["Rate"]) > 1.9
    assert min(r["Hm"]["Rate"]) > 2.9
