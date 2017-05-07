import logging

# Type dependencies
import overloaded_type
import reduced_functional


def taylor_test(J, m, h, dJdm=None, Hm=None):
    """Run a taylor test on the functional J around point m in direction h.
    
    Given a functional J, a point in control space m, and a direction in
    control space h, the function computes the taylor remainders and
    returns the convergence rate.

    Args:
        J (reduced_functional.ReducedFunctional): The functional to evaluate the taylor remainders of.
            Must be an instance of :class:`ReducedFunctional`, or something with a similar
            interface.
        m (overloaded_type.OverloadedType): The point in control space. Must be of same type as the
            control.
        h (overloaded_type.OverloadedType): The direction of perturbations. Must be of same type as
            the control.

    Returns:
        float: The smallest computed convergence rate of the tested perturbations.

    """

    Jm = J(m)
    dJdm = h._ad_dot(J.derivative()) if dJdm is None else dJdm
    Hm = 0 if Hm is None else Hm

    residuals = []
    epsilons = [0.01/2**i for i in range(4)]
    for eps in epsilons:

        perturbation = h._ad_mul(eps)
        Jp = J(m._ad_add(perturbation))

        res = abs(Jp - Jm - eps*dJdm - 0.5*eps**2*Hm)
        residuals.append(res)

    if min(residuals) < 1E-16:
        logging.warning("The taylor remainder is close to machine precision.")

    return min(convergence_rates(residuals, epsilons))


def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))
    print r
    return r

