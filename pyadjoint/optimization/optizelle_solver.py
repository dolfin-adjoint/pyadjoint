from __future__ import print_function
from .optimization_solver import OptimizationSolver
from .optimization_problem import MaximizationProblem
from . import constraints
import numpy
import math
from ..enlisting import enlist, delist
from ..misc import noannotations

from backend import *

__all__ = ['OptizelleSolver', 'OptizelleBoundConstraint']

def optizelle_callback(fun):
    """Optizelle swallows exceptions. Very useful for debugging! Let's work around this."""
    def safe_fun(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except:
            import traceback
            traceback.print_exc()
            raise

    return safe_fun

def safe_log(x):
    try:
        return math.log(x)
    except ValueError:
        return -numpy.inf

class OptizelleBoundConstraint(constraints.InequalityConstraint):
    """A class that enforces the bound constraint l <= m or m >= u."""

    def __init__(self, m, bound, type):
        assert type is 'lower' or type is 'upper'
        assert isinstance(m, (Function, Constant))
        self.m = m
        self.bound = bound

        if isinstance(self.m, Constant):
            assert hasattr(bound, '__float__')
            self.bound = float(bound)

        if type is 'lower':
            self.scale = +1.0
        else:
            self.scale = -1.0

        if not isinstance(self.bound, (float, Function)):
            raise TypeError("Your %s bound must be a Function or a Constant or a float." % type)

    def output_workspace(self):
        if isinstance(self.m, Function):
            return Function(self.m.function_space())
        elif isinstance(self.m, Constant):
            return [0.0]

    def function(self, m):
        if isinstance(self.m, Constant):
            return [self.scale*(float(m) - float(self.bound))]
        elif isinstance(self.m, Function):
            out = Function.copy(m, deepcopy=True)

            if isinstance(self.bound, float):
                out_vec = out.vector()
                out_vec *= self.scale
                out_vec[:] -= self.scale*self.bound
            elif isinstance(self.bound, Function):
                out.assign(self.scale*out - self.scale*self.bound)
            return out

    def jacobian_action(self, m, dm, result):
        if isinstance(self.m, Constant):
            result[0] = self.scale*dm[0]
        elif isinstance(self.m, Function):
            # Make sure dm is in the right PETSc state (why would it not be?)
            as_backend_type(dm.vector()).apply("")
            result.assign(self.scale*dm)

    def jacobian_adjoint_action(self, m, dp, result):
        if isinstance(self.m, Constant):
            result[0] = self.scale*dp[0]
        elif isinstance(self.m, Function):
            # Make sure dm is in the right PETSc state (why would it not be?)
            as_backend_type(dp.vector()).apply("")
            result.assign(self.scale*dp)

    def hessian_action(self, m, dm, dp, result):
        if isinstance(self.m, Constant):
            result[0] = 0.0
        elif isinstance(self.m, Function):
            result.vector().zero()

class DolfinVectorSpace(object):
    """Optizelle wants a VectorSpace object that tells it how to do the linear algebra."""

    inner_product = None

    def __init__(self, parameters):
        self.parameters = parameters

    @staticmethod
    def __deep_copy_obj(x):
        if isinstance(x, GenericFunction):
            x.vector().apply("")
            return Function.copy(x, deepcopy=True)
        elif isinstance(x, Constant):
            return Constant(float(x))
        elif isinstance(x, numpy.ndarray):
            return numpy.array(x)
        else:
            raise NotImplementedError

    @staticmethod
    def __assign_obj(x, y):
        if isinstance(x, GenericFunction):
            assert isinstance(y, GenericFunction)
            x.vector().apply("")
            y.assign(x)
        elif isinstance(x, Constant):
            y.assign(float(x))
        elif isinstance(x, numpy.ndarray):
            assert isinstance(y, numpy.ndarray)
            y[:] = x
        else:
            raise NotImplementedError

    @staticmethod
    def __scale_obj(alpha, x):
        if isinstance(x, GenericFunction):
            x.vector()[:] *= alpha
        elif isinstance(x, Constant):
            x.assign(alpha * float(x))
        elif isinstance(x, numpy.ndarray):
            x.__imul__(alpha)
        else:
            raise NotImplementedError

    @staticmethod
    def __zero_obj(x):
        if isinstance(x, GenericFunction):
            x.vector().zero()
        elif isinstance(x, Constant):
            x.assign(0.0)
        elif isinstance(x, numpy.ndarray):
            x.fill(0.0)
        else:
            raise NotImplementedError

    @staticmethod
    def __rand(x):
        if isinstance(x, GenericFunction):
            xvec = x.vector()
            xvec.set_local( numpy.random.random(xvec.local_size()) )
        elif isinstance(x, Constant):
            raise NotImplementedError
        elif isinstance(x, numpy.ndarray):
            x[:] = numpy.random.random(x.shape)
        else:
            raise NotImplementedError

    @staticmethod
    def __axpy_obj(alpha, x, y):
        if isinstance(x, GenericFunction):
            assert isinstance(y, GenericFunction)
            y.vector().axpy(alpha, x.vector())
        elif isinstance(x, Constant):
            y.assign(alpha * float(x) + float(y))
        elif isinstance(x, numpy.ndarray):
            assert isinstance(y, numpy.ndarray)
            y.__iadd__(alpha*x)
        else:
            raise NotImplementedError

    @staticmethod
    def __inner_obj(x, y):
        if isinstance(x, GenericFunction):
            assert isinstance(y, GenericFunction)

            if DolfinVectorSpace.inner_product == "H1":
                return assemble((inner(x, y) + inner(grad(x), grad(y)))*dx)
            elif DolfinVectorSpace.inner_product == "L2":
                return assemble(inner(x, y)*dx)
            elif DolfinVectorSpace.inner_product == "l2":
                return x.vector().inner(y.vector())
            else:
                raise ValueError("No inner product specified for DolfinVectorSpace")

        elif isinstance(x, Constant):
            return float(x)*float(y)
        elif isinstance(x, numpy.ndarray):
            assert isinstance(y, numpy.ndarray)
            return numpy.inner(x, y)
        else:
            raise NotImplementedError

    @staticmethod
    def __prod_obj(x, y, z):
        if isinstance(x, GenericFunction):
            assert isinstance(y, GenericFunction)
            assert isinstance(z, GenericFunction)
            z.vector()[:] = x.vector() * y.vector()
        elif isinstance(x, Constant):
            z.assign(float(x)*float(y))
        elif isinstance(x, numpy.ndarray):
            assert isinstance(y, numpy.ndarray)
            assert isinstance(z, numpy.ndarray)
            z[:] = x*y
        else:
            raise NotImplementedError

    @staticmethod
    def __id_obj(x):
        if isinstance(x, GenericFunction):
            x.assign(Constant(1.0))
        elif isinstance(x, Constant):
            x.assign(Constant(1.0))
        elif isinstance(x, numpy.ndarray):
            x.fill(1.0)
        else:
            raise NotImplementedError

    @staticmethod
    def __linv_obj(x, y, z):
        if isinstance(x, GenericFunction):
            assert isinstance(y, GenericFunction)
            assert isinstance(z, GenericFunction)
            z.vector().set_local( y.vector().array() / x.vector().array() )
            z.vector().apply("insert")
        elif isinstance(x, Constant):
            z.assign(float(y) / float(x))
        elif isinstance(x, numpy.ndarray):
            assert isinstance(y, numpy.ndarray)
            assert isinstance(z, numpy.ndarray)
            z[:] = numpy.divide(y, x)
        else:
            raise NotImplementedError

    @staticmethod
    def __barr_obj(x):
        if isinstance(x, GenericFunction):
            return assemble(ln(x)*dx)
        elif isinstance(x, Constant):
            return safe_log(float(x))
        elif isinstance(x, numpy.ndarray):
            return sum(safe_log(xx) for xx in x)
        else:
            raise NotImplementedError

    @staticmethod
    def __srch_obj(x, y):
        if isinstance(x, GenericFunction):
            assert isinstance(y, GenericFunction)
            if any(x.vector() < 0):
                my_min = min(-yy/xx for (xx, yy) in zip(x.vector().get_local(), y.vector().get_local()) if xx < 0)
            else:
                my_min = numpy.inf

            return MPI.min(x.function_space().mesh().mpi_comm(), my_min)

        elif isinstance(x, Constant):
            if float(x) < 0:
                return -float(y)/float(x)
            else:
                return numpy.inf

        elif isinstance(x, numpy.ndarray):
            assert isinstance(y, numpy.ndarray)
            if any(x < 0):
                return min(-yy/xx for (xx, yy) in zip(x, y) if xx < 0)
            else:
                return numpy.inf

        else:
            raise NotImplementedError

    @staticmethod
    def __symm_obj(x):
        pass

    @staticmethod
    @optizelle_callback
    def init(x):

        return [DolfinVectorSpace.__deep_copy_obj(xx) for xx in x]

    @staticmethod
    @optizelle_callback
    def copy(x, y):
        [DolfinVectorSpace.__assign_obj(xx, yy) for (xx, yy) in zip(x, y)]

    @staticmethod
    @optizelle_callback
    def scal(alpha, x):
        [DolfinVectorSpace.__scale_obj(alpha, xx) for xx in x]

    @staticmethod
    @optizelle_callback
    def zero(x):
        [DolfinVectorSpace.__zero_obj(xx) for xx in x]

    @staticmethod
    @optizelle_callback
    def axpy(alpha, x, y):
        [DolfinVectorSpace.__axpy_obj(alpha, xx, yy) for (xx, yy) in zip(x, y)]

    @staticmethod
    @optizelle_callback
    def innr(x, y):
        return sum(DolfinVectorSpace.__inner_obj(xx, yy) for (xx, yy) in zip(x, y))

    @staticmethod
    @optizelle_callback
    def rand(x):
        [DolfinVectorSpace.__rand(xx) for xx in x]

    @staticmethod
    @optizelle_callback
    def prod(x, y, z):
        [DolfinVectorSpace.__prod_obj(xx, yy, zz) for (xx, yy, zz) in zip(x, y, z)]

    @staticmethod
    @optizelle_callback
    def id(x):
        [DolfinVectorSpace.__id_obj(xx) for xx in x]

    @staticmethod
    @optizelle_callback
    def linv(x, y, z):
        [DolfinVectorSpace.__linv_obj(xx, yy, zz) for (xx, yy, zz) in zip(x, y, z)]

    @staticmethod
    @optizelle_callback
    def barr(x):
        return sum(DolfinVectorSpace.__barr_obj(xx) for xx in x)

    @staticmethod
    @optizelle_callback
    def srch(x,y):
        return min(DolfinVectorSpace.__srch_obj(xx, yy) for (xx, yy) in zip(x, y))

    @staticmethod
    @optizelle_callback
    def symm(x):
        [DolfinVectorSpace.__symm_obj(xx) for xx in x]

    @staticmethod
    @optizelle_callback
    def normsqdiff(x, y):
        """
        Compute ||x -y||^2; the squared norm of the difference between two functions.

        Not part of the Optizelle vector space API, but useful for us."""

        xx = DolfinVectorSpace.init(x)
        DolfinVectorSpace.axpy(-1, y, xx)
        normsq = DolfinVectorSpace.innr(xx, xx)
        return normsq

try:
    # May not have optizelle installed. That's why this is in a try block.
    import Optizelle
    import copy

    class OptizelleObjective(Optizelle.ScalarValuedFunction):

        def __init__(self, rf, scale=1):
            self.rf = rf
            self.last_x = None
            self.last_J = None
            self.scale = scale

        @optizelle_callback
        def eval(self, x):
            if self.last_x is not None:
                normsq = DolfinVectorSpace.normsqdiff(x, self.last_x)
                if normsq == 0.0:
                    return self.last_J

            self.last_x = DolfinVectorSpace.init(x)
            self.rf(x)
            self.last_J = self.scale*self.rf(x)
            return self.last_J

        def riesz_projection(self, funcs, inner_product):
            projs = []
            for func in funcs:
                if isinstance(func, Function) and inner_product!="l2":
                    V = func.function_space()
                    u = TrialFunction(V)
                    v = TestFunction(V)
                    if inner_product=="L2":
                        M = assemble(inner(u, v)*dx)
                    elif inner_product=="H1":
                        M = assemble((inner(u, v) + inner(grad(u), grad(v)))*dx)
                    else:
                        raise ValueError("Unknown inner product %s".format(inner_product))
                    proj = Function(V)
                    solve(M, proj.vector(), func.vector())
                    projs.append(proj)
                else:
                    projs.append(func)
            return projs

        @optizelle_callback
        def grad(self, x, gradient):
            self.eval(x)
            out = self.rf.derivative(forget=False, project=False)
            out = self.riesz_projection(out, DolfinVectorSpace.inner_product)
            DolfinVectorSpace.scal(self.scale, out)
            DolfinVectorSpace.copy(out, gradient)

        @optizelle_callback
        def hessvec(self, x, dx, H_dx):
            self.eval(x)
            H = enlist(self.rf.hessian(dx, project=True))
            DolfinVectorSpace.scal(self.scale, H)
            DolfinVectorSpace.copy(H, H_dx)


    class OptizelleConstraints(Optizelle.VectorValuedFunction):
        ''' This class generates a (equality and inequality) constraint object from
            a dolfin_adjoint.Constraint which is compatible with the Optizelle
            interface.
        '''

        def __init__(self, problem, constraints):
            self.constraints = constraints
            self.list_type = problem.reduced_functional.controls

        @optizelle_callback
        def eval(self, x, y):
            ''' Evaluates the constraints and stores the result in y. '''


            if self.constraints._get_constraint_dim() == 0:
                return


            x_list = delist(x, self.list_type)

            if isinstance(y, Function):
                y.assign(self.constraints.function(x_list))
            else:
                y[:] = self.constraints.function(x_list)

        @optizelle_callback
        def p(self, x, dx, y):
            ''' Evaluates the Jacobian action and stores the result in y. '''

            if self.constraints._get_constraint_dim() == 0:
                return

            x_list = delist(x, self.list_type)
            dx_list = delist(dx, self.list_type)

            self.constraints.jacobian_action(x_list, dx_list, y)

        @optizelle_callback
        def ps(self, x, dy, z):
            '''
                Evaluates the Jacobian adjoint action and stores the result in y.
                    z=g'(x)*dy
            '''

            if self.constraints._get_constraint_dim() == 0:
                return

            x_list = delist(x, self.list_type)
            z_list = delist(z, self.list_type)

            self.constraints.jacobian_adjoint_action(x_list, dy, z_list)

        @optizelle_callback
        def pps(self, x, dx, dy, z):
            '''
                Evaluates the Hessian adjoint action in directions dx and dy
                and stores the result in z.
                    z=(g''(x)dx)*dy
            '''

            if self.constraints._get_constraint_dim() == 0:
                return

            x_list = delist(x, self.list_type)
            dx_list = delist(dx, self.list_type)
            z_list = delist(z, self.list_type)

            self.constraints.hessian_action(x_list, dx_list, dy, z_list)

except ImportError:
    pass


class OptizelleSolver(OptimizationSolver):
    """
    Use optizelle to solve the given optimisation problem.

    The optizelle State instance is accessible as solver.state.
    See dir(solver.state) for the parameters that can be set,
    and the optizelle manual for details.
    """
    def __init__(self, problem, inner_product="L2", parameters=None):
        """
        Create a new OptizelleSolver.

        The argument inner_product specifies the inner product to be used for
        the control space.

        To set optizelle-specific options, do e.g.

          solver = OptizelleSolver(problem, parameters={'maximum_iterations': 100,
                                            'optizelle_parameters': {'krylov_iter_max': 100}})
        """
        try:
            import Optizelle
            import Optizelle.Unconstrained.State
            import Optizelle.Unconstrained.Functions
            import Optizelle.Unconstrained.Algorithms
            import Optizelle.Constrained.State
            import Optizelle.Constrained.Functions
            import Optizelle.Constrained.Algorithms
            import Optizelle.EqualityConstrained.State
            import Optizelle.EqualityConstrained.Functions
            import Optizelle.EqualityConstrained.Algorithms
            import Optizelle.InequalityConstrained.State
            import Optizelle.InequalityConstrained.Functions
            import Optizelle.InequalityConstrained.Algorithms
        except ImportError:
            print("Could not import Optizelle.")
            raise

        DolfinVectorSpace.inner_product = inner_product

        OptimizationSolver.__init__(self, problem, parameters)

        self.__build_optizelle_state()

    def __build_optizelle_state(self):

        # Optizelle does not support maximization problem directly,
        # hence we negate the functional instead
        if isinstance(self.problem, MaximizationProblem):
            scale = -1
        else:
            scale = +1

        bound_inequality_constraints = []
        if self.problem.bounds is not None:
            # We need to process the damn bounds
            for (control, bound) in zip(self.problem.reduced_functional.controls, self.problem.bounds):
                (lb, ub) = bound

                if lb is not None:
                    bound_inequality_constraints.append(OptizelleBoundConstraint(control.data(), lb, 'lower'))

                if ub is not None:
                    bound_inequality_constraints.append(OptizelleBoundConstraint(control.data(), ub, 'upper'))

        self.bound_inequality_constraints = bound_inequality_constraints

        # Create the appropriate Optizelle state, taking into account which
        # type of constraints we have (unconstrained, (in)-equality constraints).
        if self.problem.constraints is None:
            num_equality_constraints = 0
            num_inequality_constraints = 0 + len(bound_inequality_constraints)
        else:
            num_equality_constraints = self.problem.constraints.equality_constraints()._get_constraint_dim()
            num_inequality_constraints = self.problem.constraints.inequality_constraints()._get_constraint_dim() + len(bound_inequality_constraints)

        x = [p.data() for p in self.problem.reduced_functional.controls]

        # Unconstrained case
        if num_equality_constraints == 0 and num_inequality_constraints == 0:
            self.state = Optizelle.Unconstrained.State.t(DolfinVectorSpace, x)
            self.fns = Optizelle.Unconstrained.Functions.t()
            self.fns.f = OptizelleObjective(self.problem.reduced_functional,
                    scale=scale)

            log(INFO, "Found no constraints.")

        # Equality constraints only
        elif num_equality_constraints > 0 and num_inequality_constraints == 0:

            # Allocate memory for the equality multiplier
            equality_constraints = self.problem.constraints.equality_constraints()
            y = equality_constraints.output_workspace()

            self.state = Optizelle.EqualityConstrained.State.t(DolfinVectorSpace, DolfinVectorSpace, x, y)
            self.fns = Optizelle.Constrained.Functions.t()

            self.fns.f = OptizelleObjective(self.problem.reduced_functional,
                    scale=scale)
            self.fns.g = OptizelleConstraints(self.problem, equality_constraints)

            log(INFO, "Found no equality and %i inequality constraints." % equality_constraints._get_constraint_dim())

        # Inequality constraints only
        elif num_equality_constraints == 0 and num_inequality_constraints > 0:

            # Allocate memory for the inequality multiplier
            if self.problem.constraints is not None:
                inequality_constraints = self.problem.constraints.inequality_constraints()
                all_inequality_constraints = constraints.MergedConstraints(inequality_constraints.constraints + bound_inequality_constraints)
            else:
                all_inequality_constraints = constraints.MergedConstraints(bound_inequality_constraints)
            z = all_inequality_constraints.output_workspace()

            self.state = Optizelle.InequalityConstrained.State.t(DolfinVectorSpace, DolfinVectorSpace, x, z)
            self.fns = Optizelle.InequalityConstrained.Functions.t()

            self.fns.f = OptizelleObjective(self.problem.reduced_functional,
                    scale=scale)
            self.fns.h = OptizelleConstraints(self.problem, all_inequality_constraints)

            log(INFO, "Found no equality and %i inequality constraints." % all_inequality_constraints._get_constraint_dim())

        # Inequality and equality constraints
        else:
            # Allocate memory for the equality multiplier
            equality_constraints = self.problem.constraints.equality_constraints()
            y = equality_constraints.output_workspace()

            # Allocate memory for the inequality multiplier
            if self.problem.constraints is not None:
                inequality_constraints = self.problem.constraints.inequality_constraints()
                all_inequality_constraints = constraints.MergedConstraints(inequality_constraints.constraints + bound_inequality_constraints)
            else:
                all_inequality_constraints = constraints.MergedConstraints(bound_inequality_constraints)
            z = all_inequality_constraints.output_workspace()

            self.state = Optizelle.Constrained.State.t(DolfinVectorSpace, DolfinVectorSpace, DolfinVectorSpace, x, y, z)
            self.fns = Optizelle.Constrained.Functions.t()

            self.fns.f = OptizelleObjective(self.problem.reduced_functional,
                    scale=scale)
            self.fns.g = OptizelleConstraints(self.problem, equality_constraints)
            self.fns.h = OptizelleConstraints(self.problem, all_inequality_constraints)

            log(INFO, "Found %i equality and %i inequality constraints." % (equality_constraints._get_constraint_dim(), all_inequality_constraints._get_constraint_dim()))


        # Set solver parameters
        self.__set_optizelle_parameters()

    def __set_optizelle_parameters(self):

        if self.parameters is None: return

        # First, set the default parameters.
        if 'maximum_iterations' in self.parameters:
            self.state.iter_max = self.parameters['maximum_iterations']

        # FIXME: is there a common 'tolerance' between all solvers supported by
        # dolfin-adjoint?

        # Then set any optizelle-specific ones.
        if 'optizelle_parameters' in self.parameters:
            optizelle_parameters = self.parameters['optizelle_parameters']
            for key in optizelle_parameters:
                try:
                    setattr(self.state, key, optizelle_parameters[key])
                except AttributeError:
                    print("Error: unknown optizelle option %s." % key)
                    raise

    @noannotations
    def solve(self):
        """Solve the optimization problem and return the optimized parameters."""

        if self.problem.constraints is None:
            num_equality_constraints = 0
            num_inequality_constraints = 0 + len(self.bound_inequality_constraints)
        else:
            num_equality_constraints = self.problem.constraints.equality_constraints()._get_constraint_dim()
            num_inequality_constraints = self.problem.constraints.inequality_constraints()._get_constraint_dim() + len(self.bound_inequality_constraints)

        # No constraints
        if num_equality_constraints == 0 and num_inequality_constraints == 0:
            Optizelle.Unconstrained.Algorithms.getMin(DolfinVectorSpace, Optizelle.Messaging.stdout, self.fns, self.state)

        # Equality constraints only
        elif num_equality_constraints > 0 and num_inequality_constraints == 0:
            Optizelle.EqualityConstrained.Algorithms.getMin(DolfinVectorSpace, DolfinVectorSpace, Optizelle.Messaging.stdout, self.fns, self.state)

        # Inequality constraints only
        elif num_equality_constraints == 0 and num_inequality_constraints > 0:
            Optizelle.InequalityConstrained.Algorithms.getMin(DolfinVectorSpace, DolfinVectorSpace, Optizelle.Messaging.stdout, self.fns, self.state)

        # Inequality and equality constraints
        else:
            Optizelle.Constrained.Algorithms.getMin(DolfinVectorSpace, DolfinVectorSpace, DolfinVectorSpace, Optizelle.Messaging.stdout, self.fns, self.state)

        # Print out the reason for convergence
        # FIXME: Use logging
        print("The algorithm stopped due to: %s" % (Optizelle.OptimizationStop.to_string(self.state.opt_stop)))

        # Return the optimal control
        list_type = self.problem.reduced_functional.controls
        return delist(self.state.x, list_type)
