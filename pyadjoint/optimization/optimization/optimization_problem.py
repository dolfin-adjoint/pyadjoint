import collections

from .constraints import Constraint, canonicalise
from ..overloaded_type import OverloadedType, create_overloaded_object
from ..reduced_functional import ReducedFunctional

__all__ = ['MinimizationProblem', 'MaximizationProblem']


class OptimizationProblem(object):
    """A class that encapsulates all the information required to formulate a
    reduced optimisation problem. Don't instantiate this: instantiate
    a MinimizationProblem or a MaximizationProblem."""

    def __init__(self, reduced_functional, bounds=None, constraints=None):

        bounds = self.enlist(bounds)
        self.__check_arguments(reduced_functional, bounds, constraints)

        #: reduced_functional: a dolfin_adjoint.ReducedFunctional object that
        #: encapsulates a Function and Control
        self.reduced_functional = reduced_functional

        #: bounds: lower and upper bounds for the control (optional). None means
        #: unbounded. if not None, then it must be a list of the same length as
        #: the number controls for the reduced_functional. Each entry in the list
        #: must be a tuple (lb, ub), where ub and lb are floats, or objects
        #: of the same kind as the control.
        self.bounds = bounds

        #: constraints: general (possibly nonlinear) constraints on the controls.
        #: None means no constraints, otherwise a Constraint object or a list of
        #: Constraints.
        self.constraints = canonicalise(constraints)

    def __check_arguments(self, reduced_functional, bounds, constraints):

        if type(self) is OptimizationProblem:
            raise TypeError("Instantiate a MinimizationProblem or MaximizationProblem.")

        if not isinstance(reduced_functional, ReducedFunctional):
            raise TypeError("reduced_functional should be a ReducedFunctional")

        if bounds is not None:
            if len(bounds) != len(reduced_functional.controls):
                raise TypeError("bounds should be of length number of controls of the ReducedFunctional")
            for (bound, control) in zip(bounds, reduced_functional.controls):
                if len(bound) != 2:
                    raise TypeError("Each bound should be a tuple of length 2 (lb, ub)")

                for b in bound:
                    b = create_overloaded_object(b, suppress_warning=True)
                    klass = control.tape_value().__class__
                    if not (isinstance(b, (int, float, type(None), klass))):
                        raise TypeError("This pair (lb, ub) should be None, a float, or a %s." % klass)

        if not ((constraints is None)
                or (isinstance(constraints, Constraint))
                or (isinstance(constraints, list))):
            raise TypeError("constraints should be None or a Constraint or a list of Constraints")

    def enlist(self, bounds):
        """Make bounds into canonical format: a list (ideally the same length as the number of controls)."""
        if bounds is None:
            return None

        if not isinstance(bounds, collections.abc.Iterable):
            raise TypeError("bounds must be iterable.")

        should_i_make_a_damn_list = True

        if len(bounds) != 2:
            should_i_make_a_damn_list = False

        if len(bounds) == 2:  # support 'bounds=(lb, ub)' as well as 'bounds=[(lb, ub)]'
            for bound in bounds:
                if isinstance(bound, collections.abc.Iterable) and not isinstance(bound, OverloadedType):
                    should_i_make_a_damn_list = False

        if should_i_make_a_damn_list:
            return [bounds]
        else:
            return bounds


class MinimizationProblem(OptimizationProblem):
    """A class that encapsulates all the information required to formulate a
    reduced minimization problem."""
    pass


class MaximizationProblem(OptimizationProblem):
    """A class that encapsulates all the information required to formulate a
    reduced maximization problem."""
    pass
