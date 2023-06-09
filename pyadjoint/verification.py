import logging

from .enlisting import Enlist
from .tape import stop_annotating


def taylor_test(J, m, h, dJdm=None, Hm=0):
    """Run a taylor test on the functional J around point m in direction h.

    Given a functional J, a point in control space m, and a direction in
    control space h, the function computes the taylor remainders and
    returns the convergence rate.

    Args:
        J (reduced_functional.ReducedFunctional): The functional to evaluate the taylor remainders of.
            Must be an instance of :class:`ReducedFunctional`, or something with a similar
            interface.
        m (overloaded_type.OverloadedType): The expansion points in control space. Must be of same type as the
            control.
        h (overloaded_type.OverloadedType): The direction of perturbations. Must be of same type as
            the control.

    Returns:
        float: The smallest computed convergence rate of the tested perturbations.

    """
    with stop_annotating():
        hs = Enlist(h)
        ms = Enlist(m)

        if len(hs) != len(ms):
            raise ValueError(
                "%d perturbations are given but only %d expansion points are provided" % (len(hs), len(ms)))

        Jm = J(m)
        if dJdm is None:
            ds = Enlist(J.derivative())
            if len(ds) != len(ms):
                raise ValueError(
                    "The derivative of J depends on %d variables but only %d expansion points are given" % (
                        len(ds), len(ms)))
            dJdm = sum(hi._ad_dot(di) for hi, di in zip(hs, ds))

        def perturbe(eps):
            ret = [mi._ad_add(hi._ad_mul(eps)) for mi, hi in zip(ms, hs)]
            return ms.delist(ret)

        print("Running Taylor test")
        residuals = []
        epsilons = [0.01 / 2 ** i for i in range(4)]
        for eps in epsilons:
            Jp = J(perturbe(eps))
            res = abs(Jp - Jm - eps * dJdm - 0.5 * eps ** 2 * Hm)
            residuals.append(res)

        if min(residuals) < 1E-15:
            logging.warning("The taylor remainder is close to machine precision.")
        print("Computed residuals: {}".format(residuals))
    return min(convergence_rates(residuals, epsilons))


def convergence_rates(E_values, eps_values, show=True):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i] / E_values[i - 1])
                 / log(eps_values[i] / eps_values[i - 1]))
    if show:
        print("Computed convergence rates: {}".format(r))
    return r


def taylor_to_dict(J, m, h):
    """Run a 0th, 1st and second order taylor test on the functional J
      around point m in direction h.

    Given a functional J, a point in control space m, and a direction in
    control space h, the function computes the taylor remainders and
    returns the convergence rate in a dictionary.

    Args:
        J (reduced_functional.ReducedFunctional): The functional to evaluate the taylor remainders of.
            Must be an instance of :class:`ReducedFunctional`, or something with a similar
            interface.
        m (overloaded_type.OverloadedType): The expansion points in control space. Must be of same type as the
            control.
        h (overloaded_type.OverloadedType): The direction of perturbations. Must be of same type as
            the control.

    Returns:
        dict: The perturbation sizes, residuals and rates of the tests.

            eps (list): List of all perturbation sizes used, eps[i]*h.
            R0 (dict): Results from 0th order taylor test (finite difference).
                Residual (list): The computed residuals.
                Rate (list): The computed convergence rates based on eps and residuals. Expected to be 1.0.
            R1 (dict): Results from the 1st order taylor test.
                Residual (list): The computed residuals.
                Rate (list): The computed convergence rates based on eps and residuals. Expected to be 2.0.
            R2 (dict): Results from the 2nd order taylor test.
                Residual (list): The computed residuals.
                Rate (list): The computed convergence rates based on eps and residuals. Expected to be 3.0.

    """
    with stop_annotating():
        hs = Enlist(h)
        ms = Enlist(m)

        if len(hs) != len(ms):
            raise ValueError("{0:d} perturbations are given but only {1:d} expansion points are provided"
                             .format(len(hs), len(ms)))

        Jm = J(m)
        print("Computing derivative")
        ds = Enlist(J.derivative())
        if len(ds) != len(ms):
            raise ValueError("The derivative of J depends on {0:d} variables"
                             .format(len(ds))
                             + "but only {0:d} expansion points are given".
                             format(len(ms)))
        dJdm = sum(hi._ad_dot(di) for hi, di in zip(hs, ds))

        print("Computing Hessian")

        Hm = Enlist(J.hessian(hs))
        Hmh = sum(hi._ad_dot(hmi) for hi, hmi in zip(hs, Hm))

        def perturbe(eps):
            ret = [mi._ad_add(hi._ad_mul(eps)) for mi, hi in zip(ms, hs)]
            return ms.delist(ret)

        print("Running Taylor test")
        error_dict = {"eps": None, "R0": {"Residual": [], "Rate": None},
                      "R1": {"Residual": [], "Rate": None},
                      "R2": {"Residual": [], "Rate": None}}

        epsilons = [0.01 / 2**i for i in range(4)]
        error_dict["eps"] = epsilons
        for eps in epsilons:
            Jp = J(perturbe(eps))
            error_dict["R0"]["Residual"].append(abs(Jp - Jm))
            error_dict["R1"]["Residual"].append(abs(Jp - Jm - eps * dJdm))
            error_dict["R2"]["Residual"].append(abs(Jp - Jm - eps * dJdm - 0.5 * eps**2 * Hmh))

        for key in error_dict.keys():
            if key != "eps":
                error_dict[key]["Rate"] = convergence_rates(error_dict[key]
                                                            ["Residual"][:],
                                                            error_dict["eps"],
                                                            show=False)
    J(m)
    return error_dict
