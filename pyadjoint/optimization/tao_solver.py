from functools import wraps
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
except ModuleNotFoundError:
    petsctools = None

__all__ = [
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
        if petsctools is None:
            raise RuntimeError("petsctools not available")

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


def new_control_variable(reduced_functional, *, dual=False):
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


def valid_comm(comm):
    """
    Return a valid communicator from a user provided Comm or None.

    Args:
        comm: Optional[Any[petsc4py.PETSc.Comm,mpi4py.MPI.Comm]]

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
    FORWARD = 'forward'
    TLM = 'tlm'
    ADJOINT = 'adjoint'
    HESSIAN = 'hessian'


FORWARD = RFAction.FORWARD
TLM = RFAction.TLM
ADJOINT = RFAction.ADJOINT
HESSIAN = RFAction.HESSIAN


def check_rf_action(action):
    def check_rf_action_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.action != action:
                raise NotImplementedError(
                    f'Cannot apply {str(action)} action if {self.action=}')
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
        rf (ReducedFunctional): Defines the forward model, and used to compute operator actions.
        action (RFAction): Whether to apply the TLM, adjoint, or Hessian action.
        apply_riesz (bool): Whether to apply the riesz map before returning the
            result of the action to PETSc.
        appctx (Optional[dict]): User provided context.
        always_update_tape (bool): Whether to force reevaluation of the forward model every time
            `mult` is called. If action is HESSIAN then this will also force the adjoint model to
            be reevaluated at every call to `mult`.
        comm (Optional[petsc4py.PETSc.Comm,mpi4py.MPI.Comm]): Communicator that the rf is defined over.
    """

    def __init__(self, rf, action=HESSIAN, *,
                 apply_riesz=False, appctx=None,
                 always_update_tape=False,
                 comm=PETSc.COMM_WORLD):
        comm = valid_comm(comm)

        self.rf = rf
        self.appctx = appctx
        self.control_interface = PETScVecInterface(
            tuple(c.control for c in rf.controls),
            comm=comm)
        self.apply_riesz = apply_riesz
        if action in (ADJOINT, TLM):
            self.functional_interface = PETScVecInterface(
                rf.functional, comm=comm)

        if action == HESSIAN:  # control -> control
            self.xinterface = self.control_interface
            self.yinterface = self.control_interface

            self.x = new_control_variable(rf)
            self.mult_impl = self._mult_hessian

        elif action == ADJOINT:  # functional -> control
            self.xinterface = self.functional_interface
            self.yinterface = self.control_interface

            self.x = rf.functional._ad_copy()
            self.mult_impl = self._mult_adjoint

        elif action == TLM:  # control -> functional
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
        self.always_update_tape = always_update_tape

    @classmethod
    def update(cls, obj, x, A, P):
        ctx = A.getPythonContext()
        ctx.control_interface.from_petsc(x, ctx._m)
        ctx.update_tape_values(
            update_adjoint=(ctx.action == HESSIAN))
        ctx._shift = 0

        pctx = P.getPythonContext()
        if pctx is not ctx:
            pctx.control_interface.from_petsc(x, pctx._m)
            pctx.update_tape_values(
                update_adjoint=(pctx.action == HESSIAN))
            pctx._shift = 0

    def shift(self, A, alpha):
        self._shift += alpha

    def update_tape_values(self, *, update_adjoint=True):
        _ = self.rf(self._m)
        if update_adjoint:
            _ = self.rf.derivative(apply_riesz=False)

    def mult(self, A, x, y):
        self.xinterface.from_petsc(x, self.x)
        out = self.mult_impl(A, self.x)
        self.yinterface.to_petsc(y, out)

        if self._shift != 0:
            y.axpy(self._shift, x)

    @check_rf_action(HESSIAN)
    def _mult_hessian(self, A, x):
        if self.always_update_tape:
            self.update_tape_values(update_adjoint=True)
        return self.rf.hessian(
            x, apply_riesz=self.apply_riesz)

    @check_rf_action(TLM)
    def _mult_tlm(self, A, x):
        if self.always_update_tape:
            self.update_tape_values(update_adjoint=False)
        return self.rf.tlm(x)

    @check_rf_action(ADJOINT)
    def _mult_adjoint(self, A, x):
        if self.always_update_tape:
            self.update_tape_values(update_adjoint=False)
        return self.rf.derivative(
            adj_input=x, apply_riesz=self.apply_riesz)


def ReducedFunctionalMat(rf, action=HESSIAN, *, apply_riesz=False, appctx=None,
                         always_update_tape=False, comm=None):
    """
    PETSc.Mat to apply the action of a pyadjoint.ReducedFunctional.

    If V is the control space and U is the functional space, each action has the following map:
    Jhat : V -> U
    TLM : V -> U
    Adjoint : U* -> V*
    Hessian : V x U* -> V* | V -> V*

    Args:
        rf (ReducedFunctional): Defines the forward model, and used to compute operator actions.
        action (RFAction): Whether to apply the TLM, adjoint, or Hessian action.
        apply_riesz (bool): Whether to apply the riesz map before returning the
            result of the action to PETSc.
        appctx (Optional[dict]): User provided context.
        always_update_tape (bool): Whether to force reevaluation of the forward model every time
            `mult` is called. If action is HESSIAN then this will also force the adjoint model to
            be reevaluated at every call to `mult`.
        comm (Optional[petsc4py.PETSc.Comm,mpi4py.MPI.Comm]): Communicator that the rf is defined over.
    """
    ctx = ReducedFunctionalMatCtx(
        rf, action, appctx=appctx, apply_riesz=apply_riesz,
        always_update_tape=always_update_tape, comm=comm)

    ncol = ctx.xinterface.n
    Ncol = ctx.xinterface.N

    nrow = ctx.yinterface.n
    Nrow = ctx.yinterface.N

    mat = PETSc.Mat().createPython(
        ((nrow, Nrow), (ncol, Ncol)),
        ctx, comm=ctx.control_interface.comm)
    if action == HESSIAN:
        mat.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    mat.setUp()
    mat.assemble()
    return mat


class RieszMapMatCtx:
    def __init__(self, controls, comm=None):
        comm = valid_comm(comm)

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
        rf (AbstractReducedFunctional): Defines the forward, and used to
            compute derivative information.
        always_update_tape (bool): Whether to force reevaluation of the forward model every time
            gradient or hessian is called. If hessian is called then this will also force the
            adjoint model to be reevaluated.
    """

    def __init__(self, rf, *, always_update_tape=True):
        self._reduced_functional = rf
        self.always_update_tape = always_update_tape

    @property
    def reduced_functional(self):
        """:class:`.AbstractReducedFunctional`. Defines the forward, and used
        to compute derivative information.
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
        if self.always_update_tape:
            _ = self.reduced_functional(tuple(m_i._ad_copy() for m_i in m))
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
        if self.always_update_tape:
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

        comm = valid_comm(comm)

        rf = problem.reduced_functional
        tao_objective = TAOObjective(rf)

        vec_interface = PETScVecInterface(
            tuple(control.control for control in rf.controls), comm=comm)

        tao = PETSc.TAO().create(comm=comm)

        def objective(tao, x):
            m = new_control_variable(rf)
            vec_interface.from_petsc(x, m)
            J_val = tao_objective.objective(m)
            return J_val

        def gradient(tao, x, g):
            m = new_control_variable(rf)
            vec_interface.from_petsc(x, m)
            dJ = tao_objective.gradient(m)
            vec_interface.to_petsc(g, dJ)

        def objective_gradient(tao, x, g):
            m = new_control_variable(rf)
            vec_interface.from_petsc(x, m)
            J_val, dJ = tao_objective.objective_gradient(m)
            vec_interface.to_petsc(g, dJ)
            return J_val

        tao.setObjectiveGradient(objective_gradient)
        tao.setObjective(objective)
        tao.setGradient(gradient)

        hessian_mat = ReducedFunctionalMat(
            problem.reduced_functional, appctx=appctx,
            action=HESSIAN, comm=comm)

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
            vec_interface.to_petsc(lb_vec, lbs)
            vec_interface.to_petsc(ub_vec, ubs)
            tao.setVariableBounds(lb_vec, ub_vec)

        petsctools.set_from_options(
            tao, parameters=parameters,
            options_prefix=options_prefix)

        if tao.getType() in {PETSc.TAO.Type.LMVM, PETSc.TAO.Type.BLMVM}:
            n, N = vec_interface.n, vec_interface.N

            class InitialHessian:
                """:class:`petsc4py.PETSc.Mat` context.
                """

            class InitialHessianPreconditioner:
                """:class:`petsc4py.PETSc.PC` context.
                """
                def apply(self, pc, x, y):
                    Minv_mat.mult(x, y)

            # B_0_matrix is the initial Hessian approximation (following
            # Nocedal and Wright doi: 10.1007/978-0-387-40065-5 notation). This
            # is H0 in PETSc/TAO.

            B_0_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                  InitialHessian(), comm=comm)
            B_0_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            B_0_matrix.setUp()

            B_0_matrix_pc = PETSc.PC().createPython(
                InitialHessianPreconditioner(), comm=comm)
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
        with petsctools.inserted_options(tao):
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

        controls = self.tao_objective.reduced_functional.controls
        m = tuple(control.tape_value()._ad_copy() for control in controls)

        with petsctools.inserted_options(self.tao):
            self._vec_interface.to_petsc(self.x, m)
            self.tao.solve()
            self._vec_interface.from_petsc(self.x, m)

        if self.tao.getConvergedReason() <= 0:
            # Using the same format as Firedrake linear solver errors
            raise TAOConvergenceError(
                f"TAOSolver failed to converge after {self.tao.getIterationNumber()} iterations "
                f"with reason: {_tao_reasons[self.tao.getConvergedReason()]}")
        if isinstance(controls, Enlist):
            return controls.delist(m)
        else:
            return m
