from pyadjoint import *
from pyadjoint.reduced_function import ReducedFunction


def test_simple():
    x_1 = AdjFloat(3.0)
    x_2 = AdjFloat(5.0)
    x_3 = AdjFloat(7.0)

    y_1 = x_1*x_2
    y_2 = x_3*x_2
    y_3 = x_1*x_3
    y_4 = x_1*x_2*x_3

    func = ReducedFunction([y_1, y_2, y_3, y_4], [Control(x_1), Control(x_2), Control(x_3)])

    assert func(1.0, 1.0, 1.0) == [1.0, 1.0, 1.0, 1.0]
    assert func.adj_jac_action(1.0, 0.0, 0.0, 0.0) == [1.0, 1.0, 0.0]
    assert func.adj_jac_action(1.0, 1.0, 1.0, 1.0) == [3.0, 3.0, 3.0]

    assert func(1.0, 2.0, 3.0) == [2.0, 6.0, 3.0, 6.0]
    assert func.adj_jac_action(1.0, 0.0, 0.0, 0.0) == [2.0, 1.0, 0.0]
    assert func.adj_jac_action(0.0, 0.0, 0.0, 1.0) == [6.0, 3.0, 2.0]
    assert func.adj_jac_action(1.0, 1.0, 1.0, 1.0) == [11.0, 7.0, 5.0]

    assert func(5.0, 7.0, 13.0) == [35.0, 91.0, 65.0, 455.0]
    assert func.adj_jac_action(1.0, 0.0, 0.0, 0.0) == [7.0, 5.0, 0.0]
    assert func.adj_jac_action(0.0, 1.0, 0.0, 0.0) == [0.0, 13.0, 7.0]
    assert func.adj_jac_action(0.0, 0.0, 1.0, 0.0) == [13.0, 0.0, 5.0]
    assert func.adj_jac_action(0.0, 0.0, 0.0, 1.0) == [91.0, 65.0, 35.0]
    assert func.adj_jac_action(1.0, 1.0, 1.0, 1.0) == [111.0, 83.0, 47.0]



