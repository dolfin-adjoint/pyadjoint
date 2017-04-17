
def taylor_test(J, m, h):
    """Run a taylor test on the functional J around point m in direction h.
    
    Given a functional J, a point in control space m, and a direction in
    control space h, the function computes the taylor remainders and
    returns the convergence rate.

    Args:
        J (:obj:`ReducedFunctional`): The functional to evaluate the taylor remainders of.
            Must be an instance of :class:`ReducedFunctional`, or something with a similar
            interface.
        m (:obj:`OverloadedType`): The point in control space. Must be of same type as the
            control.
        h (:obj:`OverloadedType`): The direction of perturbations. Must be of same type as
            the control.

    Returns:
        :obj:`float`: The smallest computed convergence rate of the tested perturbations.

    """

    Jm = J(m)
    dJdm = J.derivative()

    residuals = []
    epsilons = [0.01/2**i for i in range(4)]
    for eps in epsilons:

        perturbation = h._ad_mul(eps)
        Jp = J(m._ad_add(perturbation))

        res = abs(Jp - Jm - eps*h._ad_dot(dJdm))
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
    print r
    return r

