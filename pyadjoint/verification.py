import logging

from .enlisting import Enlist
from .tape import stop_annotating
from .overloaded_type import create_overloaded_objects
from numbers import Number


def taylor_test(J, m, h, dJdm=None, Hm=0.0, v=None, epsilons=None):
    """Run a taylor test on the function J around point m in direction h.

    Given a functional/function J, a point in control space m, and a direction in
    control space h, the function computes the taylor remainders and
    returns the convergence rate.

    This function calls :func:`taylor_to_dict` and returns the minimum
    convergence rate of the results from the 2nd order test. Note that the
    argument Hm=0 by default, which means the Hessian will not be automatically
    calculated by default.

    Also, note that if you do not supply v, then the TLM action will be used
    to compute the derivative, but if you do supply v, the adjoint action
    will be used.

    Args:
        See :func:`taylor_to_dict`.

    Returns:
        float: The smallest computed convergence rate of the tested perturbations.

    """

    error_dict = taylor_to_dict(J, m, h, dJdm=dJdm, Hm=Hm, v=v, epsilons=epsilons)

    residuals = error_dict["R2"]["Residual"]
    rates = error_dict["R2"]["Rate"]

    if min(residuals) < 1e-15:
        logging.warning("The taylor remainder is close to machine precision.")
    print("Computed residuals: {}".format(residuals))
    print("Computed convergence rates: {}".format(rates))
    return min(rates)


def taylor_to_dict(J, m, h, dJdm=None, Hm=None, v=None, epsilons=None, show=False):
    """Run a 0th, 1st and 2nd order taylor test on the function J
      around point m in direction h.

    Given a function J, a point in control space m, and a direction in
    control space h, `taylor_to_dict` computes the taylor remainders and
    returns the convergence rate in a dictionary.

    Args:
        J (reduced_function.ReducedFunction): The function to
            evaluate the taylor remainders of. Must be an instance of
            :class:`ReducedFunction`, or something with a similar interface.
        m (overloaded_type.OverloadedType): The expansion points in control
            space. Must be of same type as the control.
        h (overloaded_type.OverloadedType): The direction of perturbations. Must
            be of same type as the control.

    Optional:
        dJdm: The derivative of J with respect to m in the direction h. If v is
            given or if J is a functional (in which case v=1 automatically),
            then this should be a float formed by dotting J.adj_jac_action(v)
            with h. If v is not given, this should be the same type as the
            output of J. This will be automatically calculated if it is None.
        Hm: The second derivative of J with respect to m. The Hessian will be
            ignored if this is 0, but it will be automatically calculated if
            this is None. Calculating the Hessian requires v. Hm is calculated
            by dotting h with J.hess_action(h, v).
        v: This is an adjoint direction. Must be of same type as the the output
            of J. If it is given, then dJdm is calculated using the adjoint
            Jacobian action instead of the forward Jacobian action. If J is not
            a float, it will be dotted with v to convert it to a float.

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
        hs_orig = Enlist(h)
        hs = create_overloaded_objects(hs_orig)
        ms = create_overloaded_objects(m)

        if len(hs) != len(ms):
            raise ValueError(
                "{0:d} perturbations are given but only {1:d} expansion points are provided".format(
                    len(hs), len(ms)
                )
            )

        Jm, dJdm, Hm = _get_derivatives(J, m, h, dJdm, Hm, v)

        def perturbe(eps):
            ret = [mi._ad_add(hi._ad_mul(eps)) for mi, hi in zip(ms, hs)]
            return hs_orig.delist(ret)

        print("Running Taylor test")
        error_dict = {
            "R0": {"Residual": [], "Rate": None},
            "R1": {"Residual": [], "Rate": None},
            "R2": {"Residual": [], "Rate": None},
        }

        if epsilons is None:
            epsilons = [0.01 / 2 ** i for i in range(4)]
        epsilons = Enlist(epsilons)

        Jps = []
        for eps in epsilons:
            Jp = J(perturbe(eps))
            Jp, zeroth_order, first_order, second_order = calculate_residuals(
                Jm, dJdm, Hm, Jp, eps, v=v
            )

            error_dict["R0"]["Residual"].append(zeroth_order)
            error_dict["R1"]["Residual"].append(first_order)
            error_dict["R2"]["Residual"].append(second_order)
            Jps.append(Jp)

        for key in error_dict.keys():
            error_dict[key]["Rate"] = convergence_rates(
                error_dict[key]["Residual"], epsilons, show=show
            )
        error_dict["eps"] = epsilons
        error_dict["Jm"] = Jm
        error_dict["dJdm"] = dJdm
        error_dict["Jps"] = Jps

    # Reset block variable values back to original value of m before returning.
    J(m)
    return error_dict


def _get_derivatives(J, m, h, dJdm=None, Hm=None, v=None):
    hs = create_overloaded_objects(h)
    ms = create_overloaded_objects(m)

    Jm = J(m)
    if v is not None:
        Jm = create_overloaded_objects(Jm)
        Jm = sum(
            Jm_i._ad_dot(Jm_i._ad_convert_type(v_i)) for Jm_i, v_i in zip(Jm, Enlist(v))
        )

    if dJdm is None:
        print("Computing derivative")
        if isinstance(Jm, Number) and hasattr(J, "derivative"):
            ds = Enlist(J.derivative())
            if len(ds) != len(ms):
                raise ValueError(
                    "The derivative of J depends on %d variables but only %d expansion points are given"
                    % (len(ds), len(ms))
                )
            if v is not None:
                ds = create_overloaded_objects(ds)
                ds = [di._ad_mul(v) for di in ds]
            dJdm = sum(hi._ad_dot(di) for hi, di in zip(hs, ds))
        else:
            if v is None:
                dJdm = create_overloaded_objects(J.jac_action(h))
            else:
                ds = Enlist(J.adj_jac_action(v))
                if len(ds) != len(ms):
                    raise ValueError(
                        "The derivative of J depends on %d variables but only %d expansion points are given"
                        % (len(ds), len(ms))
                    )
                dJdm = sum(hi._ad_dot(di) for hi, di in zip(hs, ds))

    if Hm is None:
        if isinstance(Jm, Number) and hasattr(J, "derivative"):
            print("Computing Hessian")
            Hms = Enlist(J.hessian(h))
        else:
            if v is None:
                raise ValueError("v must be specified to compute Hessian")
            print("Computing Hessian")
            Hms = Enlist(J.hess_action(h, v))
        Hm = sum(hi._ad_dot(hmi) for hi, hmi in zip(hs, Hms))

    return Jm, dJdm, Hm


def calculate_residuals(Jm, dJdm, Hm, Jp, eps, v=None):
    def mag(res):
        return sum(ri._ad_dot(ri) for ri in res) ** 0.5

    if v is not None:
        Jp = create_overloaded_objects(Jp)
        Jp = sum(
            Jp_i._ad_dot(Jp_i._ad_convert_type(v_i)) for Jp_i, v_i in zip(Jp, Enlist(v))
        )

    if (
        isinstance(Jp, Number)
        and isinstance(Jm, Number)
        and isinstance(dJdm, Number)
        and isinstance(Hm, Number)
    ):
        res = Jp - Jm
        zeroth_order = abs(res)

        res -= eps * dJdm
        first_order = abs(res)

        res -= 0.5 * eps ** 2 * Hm
        second_order = abs(res)
    else:
        Jm = create_overloaded_objects(Jm)
        Jp = create_overloaded_objects(Jp)

        res = [None] * len(Jp)
        for i in range(len(Jp)):
            res[i] = Jp[i]._ad_add(Jm[i]._ad_mul(-1.0))
        zeroth_order = mag(res)

        dJdm = create_overloaded_objects(dJdm)
        if dJdm is not None:
            for i in range(len(dJdm)):
                res[i] = res[i]._ad_add(dJdm[i]._ad_mul(-eps))
            first_order = mag(res)
        else:
            first_order = zeroth_order

        Hm = create_overloaded_objects(Hm)
        if Hm is not None:
            for i in range(len(Hm)):
                res[i] = res[i]._ad_add(Hm[i]._ad_mul(-0.5 * eps ** 2))
            second_order = mag(res)
        else:
            second_order = first_order

    return Jp, zeroth_order, first_order, second_order


def convergence_rates(E_values, eps_values, show=True):
    from numpy import log, inf, nan

    r = []
    for i in range(1, len(eps_values)):
        if E_values[i - 1] == 0:
            r.append(nan if E_values[i] == 0 else inf)
        else:
            r.append(
                log(E_values[i] / E_values[i - 1])
                / log(eps_values[i] / eps_values[i - 1])
            )
    if show:
        print("Computed convergence rates: {}".format(r))
    return r
