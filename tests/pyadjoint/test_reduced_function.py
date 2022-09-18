from pyadjoint import *
import numpy as np


def test_reduced_function():
    def get_y(x):
        """Adjoint Jacobian:
            [1, x[1], 0]
            [1, x[0], 2*x[1]]

            Hessian with shape (dx, dx, dy):

            Hessian[0]:
                [0, 0, 0]
                [0, 1, 0]

            Hessian[1]:
                [0, 1, 0]
                [0, 0, 2]

            [v0, v1, v2] * Hessian:
                [ [0]
                  [v1]],
                [ [v1]
                  [2*v2]]

            [v0, v1, v2] * Hessian * [h0, h1]:
                [ v1 * h1, v1 * h0 + 2 * v2 * h1]
        """

        return [
            x[0] + x[1],
            x[0] * x[1],
            x[1] * x[1],
        ]

    x = [AdjFloat(3.0), AdjFloat(5.0)]
    y = get_y(x)

    with stop_annotating():
        controls = [Control(xi) for xi in x]
        rf = ReducedFunction(y, controls)

        new_x = [2.0, 3.0]
        assert rf(new_x) == get_y(new_x)
        assert rf(x) == y

        adj_input = [1.0, 2.0, 3.0]
        adj_output = [11, 37]
        assert rf.adj_jac_action(adj_input) == adj_output

        tml_input = [-1.0, -2.0]
        tml_output = [-3.0, -6.0, -12.0]
        assert rf.jac_action(tml_input)

        hess_output = [-4, -14]
        assert rf.hess_action(tml_input, adj_input) == hess_output
