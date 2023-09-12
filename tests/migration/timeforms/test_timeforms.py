from dolfin_adjoint import *
from dolfin import *
import pytest

pytest.importorskip("dolfin")


@pytest.mark.xfail
def test_evaluation():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    adjointer.time.start(0)
    u1 = project(Constant(1), V, annotate=True, name="u1")

    adj_inc_timestep(time=1, finished=False)
    u2 = project(Constant(2), V, annotate=True, name="u2")

    adj_inc_timestep(time=2, finished=False)
    u3 = project(Constant(3), V, annotate=True, name="u3")

    adj_inc_timestep(time=3, finished=True)

    m = Control(u1)

    J = Functional(u1 * dx * dt[START_TIME])
    rf = ReducedFunctional(J, m)
    assert (1.0 - rf(u1)) < 1e-10

    J = Functional(u3 * dx * dt[FINISH_TIME])
    rf = ReducedFunctional(J, m)
    assert (3.0 - rf(u1)) < 1e-10


@pytest.mark.xfail
def test_wrong_time_type():
    dt["x"]


@pytest.mark.xfail
def test_valid_time_index():
    dt[START_TIME]
    dt[FINISH_TIME]
    dt[0]
    dt[0.4]
    dt[0.4:0.8]
