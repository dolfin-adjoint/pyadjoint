from __future__ import print_function

from functools import partial

import numpy

from . import constraints
from .optimization_problem import MaximizationProblem
from .optimization_solver import OptimizationSolver
from ..reduced_functional_numpy import ReducedFunctionalNumPy
from ..reduced_functional_numpy import gather


class _IPOptProblem:
    """API used by cyipopt for wrapping the problem"""
    def __init__(self, objective, gradient, constraints, jacobian):
        self.objective = objective
        self.gradient = gradient
        self.constraints = constraints
        self.jacobian = jacobian


class IPOPTSolver(OptimizationSolver):
    """Use the cyipopt bindings to IPOPT to solve the given optimization problem.

    The cyipopt Problem instance is accessible as solver.ipopt_problem."""

    def __init__(self, problem, parameters=None):
        OptimizationSolver.__init__(self, problem, parameters)

        self.__build_ipopt_problem()
        self.__set_parameters()

    def __build_ipopt_problem(self):
        """Build the ipopt problem from the OptimizationProblem instance."""

        from pyadjoint.ipopt import cyipopt

        self.rfn = ReducedFunctionalNumPy(self.problem.reduced_functional)

        (lb, ub) = self.__get_bounds()
        (nconstraints, fun_g, jac_g, clb, cub) = self.__get_constraints()

        # A callback that evaluates the functional and derivative.
        J = self.rfn.__call__
        dJ = partial(self.rfn.derivative, forget=False)
        nlp = cyipopt.Problem(
            n=len(ub),  # length of control vector
            lb=lb,  # lower bounds on control vector
            ub=ub,  # upper bounds on control vector
            m=nconstraints,  # number of constraints
            cl=clb,  # lower bounds on constraints
            cu=cub,  # upper bounds on constraints
            problem_obj=_IPOptProblem(
                objective=J,  # to evaluate the functional
                gradient=dJ,  # to evaluate the gradient
                constraints=fun_g,  # to evaluate the constraints
                jacobian=jac_g,  # to evaluate the constraint Jacobian
            ),
        )

        """
        if rank(self.problem.reduced_functional.mpi_comm()) > 0:
            nlp.addOption('print_level', 0)    # disable redundant IPOPT output in parallel
        else:
            nlp.addOption('print_level', 6)    # very useful IPOPT output
        """
        # TODO: Earlier the commented out code above was present.
        # Figure out how to solve parallel output cases like these in pyadjoint.
        nlp.add_option("print_level", 6)

        if isinstance(self.problem, MaximizationProblem):
            # multiply objective function by -1 internally in
            # ipopt to maximise instead of minimise
            nlp.add_option('obj_scaling_factor', -1.0)

        self.ipopt_problem = nlp

    def __get_bounds(self):
        r"""Convert the bounds into the format accepted by ipopt (two numpy arrays,
        one for the lower bound and one for the upper).

        FIXME: Do we really have to pass (-\infty, +\infty) when there are no bounds?"""

        bounds = self.problem.bounds

        if bounds is not None:
            lb_list = []
            ub_list = []  # a list of numpy arrays, one for each control

            for (bound, control) in zip(bounds, self.rfn.controls):
                general_lb, general_ub = bound  # could be float, Constant, or Function

                if isinstance(general_lb, (float, int)):
                    len_control = len(self.rfn.get_global(control))
                    lb = numpy.array([float(general_lb)] * len_control)
                else:
                    lb = self.rfn.get_global(general_lb)

                lb_list.append(lb)

                if isinstance(general_ub, (float, int)):
                    len_control = len(self.rfn.get_global(control))
                    ub = numpy.array([float(general_ub)] * len_control)
                else:
                    ub = self.rfn.get_global(general_ub)

                ub_list.append(ub)

            ub = numpy.concatenate(ub_list)
            lb = numpy.concatenate(lb_list)

        else:
            # Unfortunately you really need to specify bounds, I think?!
            ncontrols = len(self.rfn.get_controls())
            max_float = numpy.finfo(numpy.double).max
            ub = numpy.array([max_float] * ncontrols)

            min_float = numpy.finfo(numpy.double).min
            lb = numpy.array([min_float] * ncontrols)

        return (lb, ub)

    def __get_constraints(self):
        constraint = self.problem.constraints

        if constraint is None:
            # The length of the constraint vector
            nconstraints = 0

            # The bounds for the constraint
            empty = numpy.array([], dtype=float)
            clb = empty
            cub = empty

            # The constraint function, should do nothing
            def fun_g(x, user_data=None):
                return empty

            # The constraint Jacobian
            def jac_g(x, user_data=None):
                return empty

            return (nconstraints, fun_g, jac_g, clb, cub)

        else:
            # The length of the constraint vector
            nconstraints = constraint._get_constraint_dim()
            # ncontrols = len(self.rfn.get_controls())

            # The constraint function
            def fun_g(x, user_data=None):
                out = numpy.array(constraint.function(x), dtype=float)
                return out

            # The constraint Jacobian:
            # flag = True  means 'tell me the sparsity pattern';
            # flag = False means 'give me the damn Jacobian'.
            def jac_g(x, user_data=None):
                j = constraint.jacobian(x)
                out = numpy.array(gather(j), dtype=float)
                return out

            # The bounds for the constraint: by the definition of our
            # constraint type, the lower bound is always zero,
            # whereas the upper bound is either zero or infinity,
            # depending on whether it's an equality constraint or inequalityconstraint.

            clb = numpy.array([0] * nconstraints)

            def constraint_ub(c):
                if isinstance(c, constraints.EqualityConstraint):
                    return [0] * c._get_constraint_dim()
                elif isinstance(c, constraints.InequalityConstraint):
                    return [numpy.inf] * c._get_constraint_dim()

            cub = numpy.array(sum([constraint_ub(c) for c in constraint], []))

            return (nconstraints, fun_g, jac_g, clb, cub)

    _param_map = {
        'tolerance': 'tol',
        'maximum_iterations': 'max_iter',
    }

    def __set_parameters(self):
        """Set some basic parameters from the parameters dictionary that the user
        passed in, if any."""

        if self.parameters is not None:
            for param, value in self.parameters.items():
                # some parameters have a different name in ipopt
                param = self._param_map.get(param, param)
                self.ipopt_problem.add_option(param, value)

    def solve(self):
        """Solve the optimization problem and return the optimized controls."""
        guess = self.rfn.get_controls()
        results = self.ipopt_problem.solve(guess)
        new_params = [control.copy_data() for control in self.rfn.controls]
        self.rfn.set_local(new_params, results[0])

        return self.rfn.controls.delist(new_params)
