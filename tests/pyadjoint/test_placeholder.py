from pyadjoint import *
from pyadjoint.placeholder import Placeholder


def test_simple():
    a = AdjFloat(2.0)
    b = AdjFloat(3.0)

    c = a*b
    d = AdjFloat(5.0)
    p = Placeholder(d)
    e = c*d

    Jhat = ReducedFunctional(e, Control(a))
    assert Jhat(2.0) == e
    p.set_value(e)
    res = Jhat(2.0)
    assert res != e
    assert res == e*6

    assert Jhat(2.0) == e * 6 ** 2
    assert Jhat(2.0) == e * 6 ** 3
    assert Jhat(2.0) == e * 6 ** 4
    # The functional changed after last evaluation so the derivative is with the new placeholder value:
    assert Jhat.derivative() == 0.5 * e * 6 ** 5

    Jhat.optimize_tape()
    assert Jhat(2.0) == e * 6 ** 5
    p.set_value(a)
    assert Jhat(2.0) == 2.0 * 6
    assert Jhat.derivative() == 0.5 * 2.0 * 6
    assert Jhat(5.0) == 5.0 * 15
    assert Jhat.derivative() == 15
