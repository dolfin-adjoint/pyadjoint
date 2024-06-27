"""This module offers a standard interface for control constraints,
that can be used with different optimisation algorithms."""

import copy

import numpy


class Constraint(object):
    def function(self, m):
        """
        Evaluate c(m), where c(m) == 0 for equality constraints and c(m) >= 0 for inequality constraints.

        c(m) must return a numpy array or a dolfin Function or Constant.
        """

        raise NotImplementedError("Constraint.function must be supplied")

    def jacobian(self, m):
        """Returns the full Jacobian matrix as a list of vector-like objects representing the gradient of
        the constraint function with respect to the parameter m.

        The objects returned must be of the same type as m's data."""

        raise NotImplementedError("Constraint.jacobian not implemented")

    def jacobian_action(self, m, dm, result):
        """Computes the Jacobian action of c(m) in direction dm and stores the result in result. """

        raise NotImplementedError("Constraint.jacobian_action is not implemented")

    def jacobian_adjoint_action(self, m, dp, result):
        """Computes the Jacobian adjoint action of c(m) in direction dp and stores the result in result. """

        raise NotImplementedError("Constraint.jacobian_adjoint_action is not implemented")

    def hessian_action(self, m, dm, dp, result):
        """Computes the Hessian action of c(m) in direction dm and dp and stores the result in result. """

        raise NotImplementedError("Constraint.hessian_action is not implemented")

    def output_workspace(self):
        """Return an object like the output of c(m) for calculations."""

        raise NotImplementedError("Constraint.output_workspace must be supplied")

    def _get_constraint_dim(self):
        """Returns the number of constraint components."""
        workspace = self.output_workspace()

        if hasattr(workspace, "_ad_dim"):
            return workspace._ad_dim()
        return len(workspace)


class EqualityConstraint(Constraint):
    """This class represents equality constraints of the form

    c_i(m) == 0

    for 0 <= i < n, where m is the parameter.
    """


class InequalityConstraint(Constraint):
    """This class represents constraints of the form

    c_i(m) >= 0

    for 0 <= i < n, where m is the parameter.
    """


numpify = lambda x: numpy.array(x) if isinstance(x, list) else x


class MergedConstraints(Constraint):
    def __init__(self, constraints):
        self.constraints = constraints

    def function(self, m):
        return [numpify(c.function(m)) for c in self.constraints]

    def jacobian(self, m):
        return [c.jacobian(m) for c in self.constraints]

    def jacobian_action(self, m, dm, result):
        [c.jacobian_action(m, dm, result[i]) for (i, c) in enumerate(self.constraints)]

    def jacobian_adjoint_action(self, m, dp, result):
        result._ad_imul(0.0)
        tmp = copy.deepcopy(result)

        for (i, c) in enumerate(self.constraints):
            c.jacobian_adjoint_action(m, dp[i], tmp)
            result._ad_iadd(tmp)

    def hessian_action(self, m, dm, dp, result):
        result._ad_imul(0.0)
        tmp = copy.deepcopy(result)

        for (i, c) in enumerate(self.constraints):
            c.hessian_action(m, dm, dp[i], tmp)
            result._ad_iadd(tmp)

    def __iter__(self):
        return iter(self.constraints)

    def output_workspace(self):
        return [numpify(c.output_workspace()) for c in self.constraints]

    def equality_constraints(self):
        """ Filters out the equality constraints """
        constraints = [c for c in self.constraints if isinstance(c, EqualityConstraint)]
        return MergedConstraints(constraints)

    def inequality_constraints(self):
        """ Filters out the inequality constraints """
        constraints = [c for c in self.constraints if isinstance(c, InequalityConstraint)]
        return MergedConstraints(constraints)

    def _get_constraint_dim(self):
        """ Returns the number of constraint components """
        return sum([c._get_constraint_dim() for c in self.constraints])


def canonicalise(constraints):
    if constraints is None:
        return None

    if isinstance(constraints, MergedConstraints):
        return constraints

    if not isinstance(constraints, list):
        return MergedConstraints([constraints])

    else:
        return MergedConstraints(constraints)
