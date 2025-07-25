from contextlib import contextmanager
from functools import wraps
import itertools
from enum import Enum
from numbers import Complex

import numpy as np

from ..enlisting import Enlist
from ..overloaded_type import OverloadedType
from .optimization_problem import MinimizationProblem
from .optimization_solver import OptimizationSolver


try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None
try:
    import petsctools
    from petsctools import OptionsManager
except ModuleNotFoundError:
    petsctools = None

__all__ = \
    [
        "TAOConvergenceError",
        "TAOSolver"
    ]


class PETScVecInterface:
    """Interface for conversion between :class:`OverloadedType` objects and
    :class:`petsc4py.PETSc.Vec` objects.

    This uses the generic interface provided by :class:`OverloadedType`.

    Args:
        x (OverloadedType or Sequence[OverloadedType]): Defines the data
            layout.
        comm (petsc4py.PETSc.Comm or mpi4py.MPI.Comm): Communicator.
    """

    def __init__(self, x, *, comm=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        x = Enlist(x)
        if comm is None:
            comm = PETSc.COMM_WORLD
        if hasattr(comm, "tompi4py"):
            comm = comm.tompi4py()

        vecs = tuple(x_i._ad_to_petsc() for x_i in x)
        n = sum(vec.getLocalSize() for vec in vecs)
        N = sum(vec.getSize() for vec in vecs)
        _, isets = PETSc.Vec().concatenate(vecs)

        self._comm = comm
        self._n = n
        self._N = N
        self._isets = isets

    @property
    def comm(self):
        """Communicator.
        """

        return self._comm

    @property
    def n(self):
        """Total number of process local DoFs."""

        return self._n

    @property
    def N(self):
        """Total number of global DoFs."""

        return self._N

    def new_petsc(self):
        """Construct a new :class:`petsc4py.PETSc.Vec`.

        Returns:
            petsc4py.PETSc.Vec: The new :class:`petsc4py.PETSc.Vec`.
        """

        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes((self.n, self.N))
        vec.setUp()
        return vec

    def from_petsc(self, y, x):
        """Copy data from a :class:`petsc4py.PETSc.Vec` to variables.

        Args:
            y (petsc4py.PETSc.Vec): The input :class:`petsc4py.PETSc.Vec`.
            x (OverloadedType or Sequence[OverloadedType]): The output
                variables.
        """

        x = Enlist(x)
        if len(x) != len(self._isets):
            raise ValueError("Invalid length")
        for iset, x_i in zip(self._isets, x):
            y_sub = y.getSubVector(iset)
            x_i._ad_from_petsc(y_sub)
            y.restoreSubVector(iset, y_sub)

    def to_petsc(self, x, y):
        """Copy data to a :class:`petsc4py.PETSc.Vec`.

        Args:
            x (petsc4py.PETSc.Vec): The output :class:`petsc4py.PETSc.Vec`.
            y (Complex, OverloadedType or Sequence[OverloadedType]):
                Values for input variables.
        """

        y = Enlist(y)
        if len(y) != len(self._isets):
            raise ValueError("Invalid length")
        for iset, y_i in zip(self._isets, y):
            x_sub = x.getSubVector(iset)
            if isinstance(y_i, Complex):
                x_sub.set(y_i)
            elif isinstance(y_i, OverloadedType):
                y_i._ad_to_petsc(vec=x_sub)
            else:
                raise TypeError(f"Unexpected type: {type(y_i)}")
            x_sub.restoreSubVector(iset, x_sub)


def new_control_variable(reduced_functional, dual=False):
    """Return new variables suitable for storing a control value or its dual.

    Args:
        reduced_functional (ReducedFunctional): The reduced functional whose
        controls are to be copied.
        dual (bool): whether to return a dual type. If False then a primal type is returned.

    Returns:
        tuple[OverloadedType]: New variables suitable for storing a control value.
    """

    return tuple(control._ad_init_zero(dual=dual)
                 for control in reduced_functional.controls)


# Modified version of flatten_parameters function from firedrake/petsc.py,
# Firedrake master branch 57e21cc8ebdb044c1d8423b48f3dbf70975d5548, first
# added 2024-08-08
def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python3

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python3

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in parameters.items():
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            warnings.warn(("Ignoring duplicate option: %s (existing value %s, new value %s)")
                          % (option, new[option], value))
        new[option] = value
    return new


# Modified version of OptionsManager class from firedrake/petsc.py,
# Firedrake master branch 57e21cc8ebdb044c1d8423b48f3dbf70975d5548, first
# added 2024-08-08
class OptionsManager(object):

    # What appeared on the commandline, we should never clear these.
    # They will override options passed in as a dict if an
    # options_prefix was supplied.
    if PETSc is not None:
        commandline_options = frozenset(PETSc.Options().getAll())

    if PETSc is not None:
        options_object = PETSc.Options()

    count = itertools.count()

    """Mixin class that helps with managing setting petsc options.

    :arg parameters: The dictionary of parameters to use.
    :arg options_prefix: The prefix to look up items in the global
        options database (may be ``None``, in which case only entries
        from ``parameters`` will be considered.  If no trailing
        underscore is provided, one is appended.  Hence ``foo_`` and
        ``foo`` are treated equivalently.  As an exception, if the
        prefix is the empty string, no underscore is appended.

    To use this, you must call its constructor to with the parameters
    you want in the options database.

    You then call :meth:`set_from_options`, passing the PETSc object
    you'd like to call ``setFromOptions`` on.  Note that this will
    actually only call ``setFromOptions`` the first time (so really
    this parameters object is a once-per-PETSc-object thing).

    So that the runtime monitors which look in the options database
    actually see options, you need to ensure that the options database
    is populated at the time of a ``SNESSolve`` or ``KSPSolve`` call.
    Do that using the :meth:`inserted_options` context manager.

    .. code-block:: python3

       with self.inserted_options():
           self.snes.solve(...)

    This ensures that the options database has the relevant entries
    for the duration of the ``with`` block, before removing them
    afterwards.  This is a much more robust way of dealing with the
    fixed-size options database than trying to clear it out using
    destructors.

    This object can also be used only to manage insertion and deletion
    into the PETSc options database, by using the context manager.
    """
    def __init__(self, parameters, options_prefix):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        super().__init__()
        if parameters is None:
            parameters = {}
        else:
            # Convert nested dicts
            parameters = flatten_parameters(parameters)
        if options_prefix is None:
            self.options_prefix = "pyadjoint_%d_" % next(self.count)
            self.parameters = parameters
            self.to_delete = set(parameters)
        else:
            if len(options_prefix) and not options_prefix.endswith("_"):
                options_prefix += "_"
            self.options_prefix = options_prefix
            # Remove those options from the dict that were passed on
            # the commandline.
            self.parameters = {k: v for k, v in parameters.items()
                               if options_prefix + k not in self.commandline_options}
            self.to_delete = set(self.parameters)
            # Now update parameters from options, so that they're
            # available to solver setup (for, e.g., matrix-free).
            # Can't ask for the prefixed guy in the options object,
            # since that does not DTRT for flag options.
            for k, v in self.options_object.getAll().items():
                if k.startswith(self.options_prefix):
                    self.parameters[k[len(self.options_prefix):]] = v
        self._setfromoptions = False

    def set_default_parameter(self, key, val):
        """Set a default parameter value.

        :arg key: The parameter name
        :arg val: The parameter value.

        Ensures that the right thing happens cleaning up the options
        database.
        """
        k = self.options_prefix + key
        if k not in self.options_object and key not in self.parameters:
            self.parameters[key] = val
            self.to_delete.add(key)

    def set_from_options(self, petsc_obj):
        """Set up petsc_obj from the options database.

        :arg petsc_obj: The PETSc object to call setFromOptions on.

        Matt says: "Only ever call setFromOptions once".  This
        function ensures we do so.
        """
        if not self._setfromoptions:
            with self.inserted_options():
                petsc_obj.setOptionsPrefix(self.options_prefix)
                # Call setfromoptions inserting appropriate options into
                # the options database.
                petsc_obj.setFromOptions()
                self._setfromoptions = True

    @contextmanager
    def inserted_options(self):
        """Context manager inside which the petsc options database
    contains the parameters from this object."""
        try:
            for k, v in self.parameters.items():
                self.options_object[self.options_prefix + k] = v
            yield
        finally:
            for k in self.to_delete:
                del self.options_object[self.options_prefix + k]


def get_valid_comm(comm):
    """
    Return a valid communicator from a user provided (possibly null) comm.

    Args:
        comm: Any[petsc4py.PETSc.Comm,mpi4py.MPI.Comm,None]

    Returns:
        mpi4py.MPI.Comm. COMM_WORLD if `comm is None`, otherwise `comm.tompi4py()`.
    """
    if comm is None:
        comm = PETSc.COMM_WORLD
    if hasattr(comm, "tompi4py"):
        comm = comm.tompi4py()
    return comm


class RFAction(Enum):
    """
    The type of linear action that a ReducedFunctionalMat should apply.
    """
    TLM = 'tlm'
    Adjoint = 'adjoint'
    Hessian = 'hessian'


TLMAction = RFAction.TLM
AdjointAction = RFAction.Adjoint
HessianAction = RFAction.Hessian


def check_rf_action(action):
    def check_rf_action_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.action != action:
                raise NotImplementedError(
                    f'Cannot apply {str(action)} action if {self.action = }')
            return func(self, *args, **kwargs)
        return wrapper
    return check_rf_action_decorator


class ReducedFunctionalMatCtx:
    """
    PETSc.Mat Python context to apply the action of a pyadjoint.ReducedFunctional.

    If V is the control space and U is the functional space, each action has the following map:
    Jhat : V -> U
    TLM : V -> U
    Adjoint : U* -> V*
    Hessian : V x U* -> V* | V -> V*

    Args:
        rf (ReducedFunctional): Defines the forward model, and used to compute operator  actions.
        action (RFAction): Whether to apply the TLM, adjoint, or Hessian action.
        apply_riesz (bool): Whether to apply the riesz map before returning the
            result of the action to PETSc.
        appctx (Optional[dict]): User provided context.
        comm (Optional[petsc4py.PETSc.Comm,mpi4py.MPI.Comm]): Communicator that the rf is defined over.
    """

    def __init__(self, rf, action=HessianAction, *,
                 apply_riesz=False, appctx=None, comm=PETSc.COMM_WORLD):
        comm = get_valid_comm(comm)

        self.rf = rf
        self.appctx = appctx
        self.control_interface = PETScVecInterface(
            tuple(c.control for c in rf.controls),
            comm=comm)
        self.apply_riesz = apply_riesz
        if action in (AdjointAction, TLMAction):
            self.functional_interface = PETScVecInterface(
                rf.functional, comm=comm)

        if action == HessianAction:  # control -> control
            self.xinterface = self.control_interface
            self.yinterface = self.control_interface

            self.x = new_control_variable(rf)
            self.mult_impl = self._mult_hessian

        elif action == AdjointAction:  # functional -> control
            self.xinterface = self.functional_interface
            self.yinterface = self.control_interface

            self.x = rf.functional._ad_copy()
            self.mult_impl = self._mult_adjoint

        elif action == TLMAction:  # control -> functional
            self.xinterface = self.control_interface
            self.yinterface = self.functional_interface

            self.x = new_control_variable(rf)
            self.mult_impl = self._mult_tlm
        else:
            raise ValueError(
                'Unrecognised {action = }.')

        self.action = action
        self._m = new_control_variable(rf)
        self._shift = 0

    @classmethod
    def update(cls, obj, x, A, P):
        ctx = A.getPythonContext()
        ctx.control_interface.from_petsc(x, ctx._m)
        ctx.update_tape_values(update_adjoint=True)
        ctx._shift = 0

    def shift(self, A, alpha):
        self._shift += alpha

    def update_tape_values(self, update_adjoint=True):
        _ = self.rf(self._m)
        if update_adjoint:
            _ = self.rf.derivative(apply_riesz=False)

    def mult(self, A, x, y):
        self.xinterface.from_petsc(x, self.x)
        out = self.mult_impl(A, self.x)
        self.yinterface.to_petsc(y, out)

        if self._shift != 0:
            y.axpy(self._shift, x)

    @check_rf_action(action=HessianAction)
    def _mult_hessian(self, A, x):
        # self.update_tape_values(update_adjoint=True)
        return self.rf.hessian(
            x, apply_riesz=self.apply_riesz)

    @check_rf_action(TLMAction)
    def _mult_tlm(self, A, x):
        # self.update_tape_values(update_adjoint=False)
        return self.rf.tlm(x)

    @check_rf_action(AdjointAction)
    def _mult_adjoint(self, A, x):
        # self.update_tape_values(update_adjoint=False)
        return self.rf.derivative(
            adj_input=x, apply_riesz=self.apply_riesz)


def ReducedFunctionalMat(rf, action=HessianAction, *, apply_riesz=False, appctx=None, comm=None):
    """
    PETSc.Mat to apply the action of a pyadjoint.ReducedFunctional.

    If V is the control space and U is the functional space, each action has the following map:
    Jhat : V -> U
    TLM : V -> U
    Adjoint : U* -> V*
    Hessian : V x U* -> V* | V -> V*

    Args:
        rf (ReducedFunctional): Defines the forward model, and used to compute operator  actions.
        action (RFAction): Whether to apply the TLM, adjoint, or Hessian action.
        apply_riesz (bool): Whether to apply the riesz map before returning the
            result of the action to PETSc.
        appctx (Optional[dict]): User provided context.
        comm (Optional[petsc4py.PETSc.Comm,mpi4py.MPI.Comm]): Communicator that the rf is defined over.
    """
    ctx = ReducedFunctionalMatCtx(
        rf, action, appctx=appctx, apply_riesz=apply_riesz, comm=comm)

    ncol = ctx.xinterface.n
    Ncol = ctx.xinterface.N

    nrow = ctx.yinterface.n
    Nrow = ctx.yinterface.N

    mat = PETSc.Mat().createPython(
        ((nrow, Nrow), (ncol, Ncol)),
        ctx, comm=ctx.control_interface.comm)
    if action == HessianAction:
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    mat.setUp()
    mat.assemble()
    return mat


class RieszMapMatCtx:
    def __init__(self, controls, comm=None):
        comm = get_valid_comm(comm)

        self.controls = Enlist(controls)
        self.vec_interface = PETScVecInterface(
            tuple(c.control for c in controls),
            comm=comm)

        self.dJ = tuple(c._ad_init_zero(dual=True)
                        for c in self.controls)

    def mult(self, mat, x, y):
        self.vec_interface.from_petsc(x, self.dJ)
        dJ = tuple(c._ad_convert_riesz(dJi, riesz_map=c.riesz_map)
                   for c, dJi in zip(self.controls, self.dJ))
        self.vec_interface.to_petsc(y, dJ)


def RieszMapMat(controls, symmetric=True, comm=None):
    ctx = RieszMapMatCtx(controls, comm=comm)

    n = ctx.vec_interface.n
    N = ctx.vec_interface.N

    mat = PETSc.Mat().createPython(
        ((n, N), (n, N)), ctx,
        comm=ctx.vec_interface.comm)
    if symmetric:
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    mat.setUp()
    mat.assemble()
    return mat


class TAOObjective:
    """Utility class for computing functional values and associated derivatives.

    Args:
        rf (ReducedFunctional): Defines the forward, and used to compute
            derivative information.
    """

    def __init__(self, rf):
        self._reduced_functional = rf

    @property
    def reduced_functional(self):
        """:class:`.ReducedFunctional`. Defines the forward, and used to
        compute derivative information.
        """

        return self._reduced_functional

    def objective(self, m):
        """Evaluate the forward.

        Args:
            m (OverloadedType or Sequence[OverloadedType]): Defines the control
                value.
        Returns:
            AdjFloat: The value of the functional.
        """

        m = Enlist(m)
        J = self.reduced_functional(tuple(m_i._ad_copy() for m_i in m))
        return J

    def gradient(self, m):
        """Compute a first derivative.

        Args:
            m (OverloadedType or Sequence[OverloadedType]): Defines the control
                value.
        Returns:
            AdjFloat: The value of the functional.
            OverloadedType or Sequence[OverloadedType]: The (dual space)
                derivative.
        """

        m = Enlist(m)
        # J = self.reduced_functional(tuple(m_i._ad_copy() for m_i in m))
        dJ = self.reduced_functional.derivative()
        return m.delist(dJ)

    def objective_gradient(self, m):
        """Evaluate the forward, and compute a first derivative.

        Args:
            m (OverloadedType or Sequence[OverloadedType]): Defines the control
                value.
        Returns:
            AdjFloat: The value of the functional.
            OverloadedType or Sequence[OverloadedType]: The (dual space)
                derivative.
        """

        m = Enlist(m)
        J = self.reduced_functional(tuple(m_i._ad_copy() for m_i in m))
        dJ = self.reduced_functional.derivative()
        return J, m.delist(dJ)

    def hessian(self, m, m_dot):
        """Evaluate the forward, and compute a second derivative action on a
        given direction.

        Args:
            m (OverloadedType or Sequence[OverloadedType]): Defines the control
                value.
            m_dot (OverloadedType or Sequence[OverloadedType]): Defines the
                direction.
        Returns:
            OverloadedType or Sequence[OverloadedType]: The (dual space)
                second derivative action on the given direction.
        """

        m = Enlist(m)
        m_dot = Enlist(m_dot)
        _ = self.reduced_functional(tuple(m_i._ad_copy() for m_i in m))
        _ = self.reduced_functional.derivative()
        ddJ = self.reduced_functional.hessian(tuple(m_dot_i._ad_copy() for m_dot_i in m_dot))
        return m.delist(ddJ)


class TAOConvergenceError(Exception):
    """Raised if a TAO solve fails to converge.
    """


if PETSc is None:
    _tao_reasons = {}
else:
    # Same approach as in _make_reasons in firedrake/solving_utils.py,
    # Firedrake master branch 57e21cc8ebdb044c1d8423b48f3dbf70975d5548
    _tao_reasons = {getattr(PETSc.TAO.Reason, key): key
                    for key in dir(PETSc.TAO.Reason)
                    if not key.startswith("_")}


class TAOSolver(OptimizationSolver):
    """Use TAO to solve an optimization problem.

    Args:
        problem (MinimizationProblem): Defines the optimization problem to be solved.
        parameters (Mapping): TAO options.
        options_prefix (Optional[str]): prefix for the TAO solver.
        appctx (Optional[dict]): User provided context.
        Pmat (Optional petsc4py.PETSc.Mat): Hessian preconditioning matrix.
        comm (petsc4py.PETSc.Comm or mpi4py.MPI.Comm): Communicator.
    """

    def __init__(self, problem, parameters, *,
                 options_prefix=None, appctx=None,
                 Pmat=None, comm=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")
        if petsctools is None:
            raise RuntimeError("petsctools not available")

        if not isinstance(problem, MinimizationProblem):
            raise TypeError("MinimizationProblem required")
        if problem.constraints is not None:
            raise NotImplementedError("Constraints not implemented")

        comm = get_valid_comm(comm)

        rf = problem.reduced_functional
        tao_objective = TAOObjective(rf)

        vec_interface = PETScVecInterface(
            tuple(control.control for control in rf.controls), comm=comm)

        to_petsc, from_petsc = vec_interface.to_petsc, vec_interface.from_petsc

        tao = PETSc.TAO().create(comm=comm)

        def objective(tao, x, g):
            m = new_control_variable(rf)
            from_petsc(x, m)
            J_val = tao_objective.objective(m)
            return J_val

        def gradient(tao, x, g):
            m = new_control_variable(rf)
            from_petsc(x, m)
            dJ = tao_objective.gradient(m)
            to_petsc(g, dJ)

        def objective_gradient(tao, x, g):
            m = new_control_variable(rf)
            from_petsc(x, m)
            J_val, dJ = tao_objective.objective_gradient(m)
            to_petsc(g, dJ)
            return J_val

        tao.setObjectiveGradient(objective_gradient)
        tao.setObjective(objective)
        tao.setGradient(gradient)

        hessian_mat = ReducedFunctionalMat(
            problem.reduced_functional, appctx=appctx,
            action=HessianAction, comm=comm)

        tao.setHessian(
            hessian_mat.getPythonContext().update,
            H=hessian_mat, P=Pmat or hessian_mat)

        Minv_mat = RieszMapMat(rf.controls, comm=comm)
        tao.setGradientNorm(Minv_mat)

        if problem.bounds is not None:
            lbs = []
            ubs = []
            assert len(problem.bounds) == len(problem.reduced_functional.controls)
            for (lb, ub), control in zip(problem.bounds, tao_objective.reduced_functional.controls):
                if lb is None:
                    lb = np.finfo(PETSc.ScalarType).min
                if ub is None:
                    ub = np.finfo(PETSc.ScalarType).max
                lbs.append(lb)
                ubs.append(ub)

            lb_vec = vec_interface.new_petsc()
            ub_vec = vec_interface.new_petsc()
            to_petsc(lb_vec, lbs)
            to_petsc(ub_vec, ubs)
            tao.setVariableBounds(lb_vec, ub_vec)

        self.options = OptionsManager(parameters, options_prefix)
        self.options.set_from_options(tao)

        if tao.getType() in {PETSc.TAO.Type.LMVM, PETSc.TAO.Type.BLMVM}:
            n, N = vec_interface.n, vec_interface.N

            class InitialHessian:
                """:class:`petsc4py.PETSc.Mat` context.
                """

            class InitialHessianPreconditioner:
                """:class:`petsc4py.PETSc.PC` context.
                """

                def apply(self, pc, x, y):
                    dJ = new_control_variable(rf, dual=True)
                    from_petsc(x, dJ)
                    assert len(tao_objective.reduced_functional.controls) == len(dJ)
                    dJ = tuple(control._ad_convert_riesz(dJ_i, riesz_map=control.riesz_map)
                               for control, dJ_i in zip(tao_objective.reduced_functional.controls, dJ))
                    to_petsc(y, dJ)

            # B_0_matrix is the initial Hessian approximation (following
            # Nocedal and Wright doi: 10.1007/978-0-387-40065-5 notation). This
            # is H0 in PETSc/TAO.

            B_0_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                  InitialHessian(), comm=comm)
            B_0_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            B_0_matrix.setUp()

            B_0_matrix_pc = PETSc.PC().createPython(InitialHessianPreconditioner(),
                                                    comm=comm)
            B_0_matrix_pc.setOperators(B_0_matrix)
            B_0_matrix_pc.setUp()

            tao.setLMVMH0(B_0_matrix)
            ksp = tao.getLMVMH0KSP()
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.setTolerances(rtol=0.0, atol=0.0, divtol=None, max_it=1)
            ksp.setPC(B_0_matrix_pc)
            ksp.setUp()
        else:
            B_0_matrix = None
            B_0_matrix_pc = None

        x = vec_interface.new_petsc()
        tao.setSolution(x)
        tao.setUp()

        super().__init__(problem, parameters)
        self._tao_objective = tao_objective
        self._vec_interface = vec_interface
        self._tao = tao
        self._x = x

    @property
    def tao_objective(self):
        """The :class:`.TAOObjective` used for the optimization.
        """

        return self._tao_objective

    @property
    def tao(self):
        """The :class:`petsc4py.PETSc.TAO` used for the optimization.
        """

        return self._tao

    @property
    def x(self):
        """The :class:`petsc4py.PETSc.Vec` used to store the solution to the
        optimization problem.
        """

        return self._x

    def solve(self):
        """Solve the optimization problem.

        Returns:
            OverloadedType or Sequence[OverloadedType]: The solution.
        """

        m = tuple(
            control.tape_value()._ad_copy()
            for control in self.tao_objective.reduced_functional.controls)
        with self.options.inserted_options():
            self._vec_interface.to_petsc(self.x, m)
            self.tao.solve()
            self._vec_interface.from_petsc(self.x, m)
        if self.tao.getConvergedReason() <= 0:
            # Using the same format as Firedrake linear solver errors
            raise TAOConvergenceError(
                f"TAOSolver failed to converge after {self.tao.getIterationNumber()} iterations "
                f"with reason: {_tao_reasons[self.tao.getConvergedReason()]}")
        return self.tao_objective.reduced_functional.controls.delist(m)
