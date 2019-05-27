from math import log
from numpy.testing import assert_approx_equal
from pyadjoint import *


def test_mul():
    a = AdjFloat(3.0)
    b = AdjFloat(5.0)
    J = a * b * a * b * a * b

    Jhat1 = ReducedFunctional(J, Control(a))
    Jhat2 = ReducedFunctional(J, Control(b))
    Jhat3 = ReducedFunctional(J, [Control(a), Control(b)])

    res = taylor_to_dict(Jhat1, a, AdjFloat(1.0))
    print(res)
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9

    res = taylor_to_dict(Jhat2, b, AdjFloat(1.0))
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9

    res = taylor_to_dict(Jhat3, [a, b], [AdjFloat(0.3), AdjFloat(0.2)])
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9


def test_pow():
    a = AdjFloat(3.0)
    b = AdjFloat(5.0)
    J = (a ** 2 * b ** 2)**2

    Jhat1 = ReducedFunctional(J, Control(a))
    Jhat2 = ReducedFunctional(J, Control(b))
    Jhat3 = ReducedFunctional(J, [Control(a), Control(b)])

    res = taylor_to_dict(Jhat1, a, AdjFloat(1.0))
    print(res)
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9

    res = taylor_to_dict(Jhat2, b, AdjFloat(1.0))
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9

    res = taylor_to_dict(Jhat3, [a, b], [AdjFloat(0.3), AdjFloat(0.2)])
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9


def test_div():
    a = AdjFloat(3.0)
    b = AdjFloat(5.0)
    J = (a ** 2 / b ** 2)**2 / (a - b)

    get_working_tape().visualise("test.dot")

    Jhat1 = ReducedFunctional(J, Control(a))
    Jhat2 = ReducedFunctional(J, Control(b))
    Jhat3 = ReducedFunctional(J, [Control(a), Control(b)])

    res = taylor_to_dict(Jhat1, a, AdjFloat(1.0))
    print(res)
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9

    res = taylor_to_dict(Jhat2, b, AdjFloat(1.0))
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9

    res = taylor_to_dict(Jhat3, [a, b], [AdjFloat(0.3), AdjFloat(0.2)])
    assert min(res["FD"]["Rate"]) > 0.9
    assert min(res["dJdm"]["Rate"]) > 1.9
    assert min(res["Hm"]["Rate"]) > 2.9
