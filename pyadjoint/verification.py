
# TODO: This function is subject to change, and so in particular
# the types given in the docstring below are not entirely accurate.
def taylor_test(J, m, h, dot=None):
    """Run a taylor test on the functional J around point m in direction h.
    
    Given a functional J, a point in control space m, and a direction in
    control space h, the function computes the taylor remainders and
    returns the convergence rate.

    Args:
        J (:obj:`ReducedFunctional`): The functional to evaluate the taylor remainders of.
            Must be an instance of :class:`ReducedFunctional`, or something with a similar
            `__call__` routine.
        m (:obj:`OverloadedType`): The point in control space.
        h (:obj:`OverloadedType`): The direction of pertubations.
        dot (func): A function to compute the inner product of the derivative and the direction.
    
    Returns:
        :obj:`float`: The smallest computed convergence rate of the tested pertubations.

    """
    if dot is None:
        from numpy import dot
    
    Jm = J(m)
    dJdm = J.derivative()

    residuals = []
    epsilons = [0.01/2**i for i in range(4)]
    for eps in epsilons:
        # TODO: Another way of solving this is to do something similar to the user-defined dot argument.
        # That is to supply a function that adds pertubations.
        # Instead at the moment the proposed solution is to let the `adj_update_value` methods
        # of the OverloadedType objects handle it. The pros of that being the flexibility in
        # how the user decides to change the control values.
        # The downside is all the extra computations needed to essentially just undo what
        # dolfin/ufl does. Which seems counterproductive.
        Jp = J(m + h*eps)

        res = abs(Jp - Jm - eps*dot(h, dJdm))
        residuals.append(res)

    # TODO: A warning if residuals are close to machine precision.
    # We would first need some error/warning messages.
    # Old dolfin-adjoint uses backend for warning/info,
    # but pyadjoint has no concept of the backend so it needs its own.

    return min(convergence_rates(residuals, epsilons))


def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))

    return r

