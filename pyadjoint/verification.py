import logging
from .tape import stop_annotating

# Type dependencies
from . import overloaded_type
from . import reduced_functional


def taylor_test(J, m, h, dJdm=None, Hm=0):
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
    if len(J.controls) > 1 and not isinstance(m, (list, tuple)):
        raise ValueError("J depends on %d controls but m is not a list or tuple."%len(J.controls))
    if len(J.controls) > 1 and not isinstance(h, (list, tuple)):
        raise ValueError("J depends on %d controls but h is not a list or tuple."%len(J.controls))

    if len(J.controls) == 1:
        def derivative(): return h._ad_dot(J.derivative())
        def perturbe(eps): return m._ad_add(h._ad_mul(eps))
    else:
        def derivative(): return sum(hi._ad_dot(di) for hi,di in zip(h,J.derivative()))
        def perturbe(eps): return [mi._ad_add(hi._ad_mul(eps)) for mi,hi in zip(m,h)]

    print("Running Taylor test")
    with stop_annotating():
        Jm = J(m)
        if dJdm is None: dJdm = derivative()

        residuals = []
        epsilons = [0.01/2**i for i in range(4)]
        for eps in epsilons:
            Jp = J(perturbe(eps))
            res = abs(Jp - Jm - eps*dJdm - 0.5*eps**2*Hm)
            residuals.append(res)

        if min(residuals) < 1E-15:
            logging.warning("The taylor remainder is close to machine precision.")
        print("Computed residuals: {}".format(residuals))
    return min(convergence_rates(residuals, epsilons))


def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))
    print("Computed convergence rates: {}".format(r))
    return r

