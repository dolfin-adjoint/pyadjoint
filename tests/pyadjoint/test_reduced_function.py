from pyadjoint import *
import numpy as np

def example1_get_y(x):
    """Adjoint Jacobian:
        [1, x1, x1**2]
        [1, x0, 2*x0*x1]

        Hessian with shape (dx, dx, dy):

        Hessian[0]:
            [0, 0, 0]
            [0, 1, 2*x1]

        Hessian[1]:
            [0, 1, 2*x1]
            [0, 0, 2*x0]

        [v0, v1, v2] * Hessian:
            [ [0]
              [v1 + 2*v2*x1]],
            [ [v1 + 2*v2*x1]]
              [2*v2*x0]]

        [v0, v1, v2] * Hessian * [h0, h1]:
            [ (v1 + 2*v2*x1) * h1, (v1 + 2*v2*x1) * h0 + 2*v2*x0*h1]
    """

    return [
        x[0] + x[1],
        x[0] * x[1],
        x[1] * x[1] * x[0],
    ]

def test_reduced_function():
    x = [AdjFloat(3.0), AdjFloat(5.0)]
    y = example1_get_y(x)

    with stop_annotating():
        controls = [Control(xi) for xi in x]
        rf = ReducedFunction(y, controls)

        new_x = [2.0, 3.0]
        assert rf(new_x) == example1_get_y(new_x)
        assert rf(x) == y

        adj_input = [1.0, 2.0, 3.0]
        adj_output = [86, 97]
        assert rf.adj_jac_action(adj_input) == adj_output

        tml_input = [-1.0, -2.0]
        tml_output = [-3.0, -11.0, -85.0]
        assert rf.jac_action(tml_input) == tml_output

        hess_output = [-64, -68]
        assert rf.hess_action(tml_input, adj_input) == hess_output

def test_taylor_test():
    x = [AdjFloat(3.0), AdjFloat(5.0)]
    y = example1_get_y(x)

    with stop_annotating():
        controls = [Control(xi) for xi in x]
        rf = ReducedFunction(y, controls)

        v = [1.0, 2.0, 3.0]
        h = [-1.0, -2.0]

        jac = rf.jac_action(h)
        adj_jac = rf.adj_jac_action(v)
        hess = rf.hess_action(h, v)

        rate_keys = ["R0", "R1", "R2"]
        results = taylor_to_dict(rf, x, h, v=v)
        for order, k in enumerate(rate_keys):
            assert min(results[k]["Rate"]) > order + 0.9, results[k]

        # Call with or without v to separately test adjoint and TLM.
        assert taylor_test(rf, x, h) > 1.9
        assert taylor_test(rf, x, h, v=v) > 1.9

        # Call with dJdm = 0. to skip derivative calculations.
        assert taylor_test(rf, x, h, dJdm=0, v=v) > 0.9
        assert taylor_test(rf, x, h, dJdm=0) > 0.9

        # Call with correct dJdm to skip derivative calculations.
        dJdm = jac
        assert taylor_test(rf, x, h, dJdm=dJdm) > 1.9

        dJdm = sum(vi * jac_i for vi, jac_i in zip(v, jac))
        assert taylor_test(rf, x, h, dJdm=dJdm, v=v) > 1.9

        dJdm = sum(hi * adj_i for hi, adj_i in zip(h, adj_jac))
        assert taylor_test(rf, x, h, dJdm=dJdm, v=v) > 1.9

        # Call with Hm = None to include Hessian calculations.
        assert taylor_test(rf, x, h, v=v, Hm=None) > 2.9

        # Call with correct Hm to skip Hessian calculations.
        Hm = sum(hi * hess_i for hi, hess_i in zip(h, hess))
        assert taylor_test(rf, x, h, Hm=Hm, v=v) > 2.9

