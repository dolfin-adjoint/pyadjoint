from __future__ import print_function
from .optimization_solver import OptimizationSolver
from .optimization_problem import MaximizationProblem
from ..reduced_functional_numpy import ReducedFunctionalNumPy
from . import constraints

from ..reduced_functional_numpy import gather
import numpy

class IPOPTSolver(OptimizationSolver):
    """Use the pyipopt bindings to IPOPT to solve the given optimization problem.

    The pyipopt Problem instance is accessible as solver.pyipopt_problem."""

    def __init__(self, problem, parameters=None):
        try:
            import pyipopt
        except ImportError:
            print("You need to install pyipopt. It is recommended to install IPOPT with HSL support!")
            raise

        OptimizationSolver.__init__(self, problem, parameters)

        self.__build_pyipopt_problem()
        self.__set_parameters()

    def __build_pyipopt_problem(self):
        """Build the pyipopt problem from the OptimizationProblem instance."""

        import pyipopt
        from functools import partial

        self.rfn = ReducedFunctionalNumPy(self.problem.reduced_functional)
        ncontrols = len(self.rfn.get_controls())

        (lb, ub) = self.__get_bounds()
        (nconstraints, fun_g, jac_g, clb, cub) = self.__get_constraints()
        constraints_nnz = nconstraints * ncontrols

        # A callback that evaluates the functional and derivative.
        J = self.rfn.__call__
        dJ = partial(self.rfn.derivative, forget=False)

        nlp = pyipopt.create(len(ub),           # length of control vector
                             lb,                # lower bounds on control vector
                             ub,                # upper bounds on control vector
                             nconstraints,      # number of constraints
                             clb,               # lower bounds on constraints,
                             cub,               # upper bounds on constraints,
                             constraints_nnz,   # number of nonzeros in the constraint Jacobian
                             0,                 # number of nonzeros in the Hessian
                             J,                 # to evaluate the functional
                             dJ,                # to evaluate the gradient
                             fun_g,             # to evaluate the constraints
                             jac_g)             # to evaluate the constraint Jacobian

        pyipopt.set_loglevel(1)                 # turn off annoying pyipopt logging

        """
        if rank(self.problem.reduced_functional.mpi_comm()) > 0:
            nlp.int_option('print_level', 0)    # disable redundant IPOPT output in parallel
        else:
            nlp.int_option('print_level', 6)    # very useful IPOPT output
        """
        # TODO: Earlier the commented out code above was present.
        # Figure out how to solve parallel output cases like these in pyadjoint.
        nlp.int_option("print_level", 6)

        if isinstance(self.problem, MaximizationProblem):
            # multiply objective function by -1 internally in
            # ipopt to maximise instead of minimise
            nlp.num_option('obj_scaling_factor', -1.0)

        self.pyipopt_problem = nlp

    def __get_bounds(self):
        """Convert the bounds into the format accepted by pyipopt (two numpy arrays,
        one for the lower bound and one for the upper).

        FIXME: Do we really have to pass (-\infty, +\infty) when there are no bounds?"""

        bounds = self.problem.bounds

        if bounds is not None:
            lb_list = []
            ub_list = [] # a list of numpy arrays, one for each control

            for (bound, control) in zip(bounds, self.rfn.controls):
                general_lb, general_ub = bound  # could be float, Constant, or Function

                if isinstance(general_lb, (float, int)):
                    len_control = len(self.rfn.get_global(control))
                    lb = numpy.array([float(general_lb)]*len_control)
                else:
                    lb = self.rfn.get_global(general_lb)

                lb_list.append(lb)

                if isinstance(general_ub, (float, int)):
                    len_control = len(self.rfn.get_global(control))
                    ub = numpy.array([float(general_ub)]*len_control)
                else:
                    ub = self.rfn.get_global(general_ub)

                ub_list.append(ub)

            ub = numpy.concatenate(ub_list)
            lb = numpy.concatenate(lb_list)

        else:
            # Unfortunately you really need to specify bounds, I think?!
            ncontrols = len(self.rfn.get_controls())
            max_float = numpy.finfo(numpy.double).max
            ub = numpy.array([max_float]*ncontrols)

            min_float = numpy.finfo(numpy.double).min
            lb = numpy.array([min_float]*ncontrols)

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
            def jac_g(x, flag, user_data=None):
                if flag:
                    rows = numpy.array([], dtype=int)
                    cols = numpy.array([], dtype=int)
                    return (rows, cols)
                else:
                    return empty

            return (nconstraints, fun_g, jac_g, clb, cub)

        else:
            # The length of the constraint vector
            nconstraints = constraint._get_constraint_dim()
            ncontrols = len(self.rfn.get_controls())

            # The constraint function
            def fun_g(x, user_data=None):
                out = numpy.array(constraint.function(x), dtype=float)
                return out

            # The constraint Jacobian:
            # flag = True  means 'tell me the sparsity pattern';
            # flag = False means 'give me the damn Jacobian'.
            def jac_g(x, flag, user_data=None):
                if flag:
                    # FIXME: Don't have any sparsity information on constraints;
                    # pass in a dense matrix (it usually is anyway).
                    rows = []
                    for i in range(nconstraints):
                        rows += [i] * ncontrols
                    cols = list(range(ncontrols)) * nconstraints
                    return (numpy.array(rows), numpy.array(cols))
                else:
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

    def __set_parameters(self):
        """Set some basic parameters from the parameters dictionary that the user
        passed in, if any."""

        if self.parameters is not None:
            for param in self.parameters:
                if param == "tolerance":
                    tol = self.parameters['tolerance']
                    self.pyipopt_problem.num_option('tol', tol)
                elif param == "maximum_iterations":
                    maxiter = self.parameters['maximum_iterations']
                    self.pyipopt_problem.int_option('max_iter', maxiter)
                else:
                    out = self.parameters[param]
                    if isinstance(out, int):
                        self.pyipopt_problem.int_option(param, out)
                    elif isinstance(out, str):
                        self.pyipopt_problem.str_option(param, out)
                    elif isinstance(out, float):
                        self.pyipopt_problem.num_option(param, out)
                    else:
                        raise ValueError("Don't know how to deal with parameter %s (a %s)" % (param, out.__class__))

    def solve(self):
        """Solve the optimization problem and return the optimized controls."""
        guess = self.rfn.get_controls()
        results = self.pyipopt_problem.solve(guess)
        new_params = [control.copy_data() for control in self.rfn.controls]
        self.rfn.set_local(new_params, results[0])

        return self.rfn.controls.delist(new_params)
