from functools import cached_property
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


class TAOObjective:
    """Utility class for computing functional values and associated
    derivatives.

    Args:
        rf (AbstractReducedFunctional): Defines the forward, and used to
            compute derivative information.
    """

    def __init__(self, rf):
        self._reduced_functional = rf

    @property
    def reduced_functional(self):
        """:class:`.AbstractReducedFunctional`. Defines the forward, and used
        to compute derivative information.
        """

        return self._reduced_functional

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

    def new_control_variable(self):
        """Return new variables suitable for storing a control value.

        Returns:
            tuple[OverloadedType]: New variables suitable for storing a control
                value.
        """

        return tuple(control._ad_init_zero(dual=False)
                     for control in self.reduced_functional.controls)

    def new_dual_control_variable(self):
        """Return new variables suitable for storing a value for a (dual space)
        derivative of the functional with respect to the control.

        Returns:
            tuple[OverloadedType]: New variables suitable for storing a value
                for a (dual space) derivative of the functional with respect to
                the control.
        """

        return tuple(control._ad_init_zero(dual=True)
                     for control in self.reduced_functional.controls)


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
        problem (MinimizationProblem): Defines the optimization problem to be
            solved.
        parameters (Mapping): TAO options.
        comm (petsc4py.PETSc.Comm or mpi4py.MPI.Comm): Communicator.
        convert_options (Mapping): Defines the `options` argument to
            :meth:`OverloadedType._ad_convert_type`.
    """

    def __init__(self, problem, parameters, *, comm=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")
        if petsctools is None:
            raise RuntimeError("petsctools not available")

        if not isinstance(problem, MinimizationProblem):
            raise TypeError("MinimizationProblem required")
        if problem.constraints is not None:
            raise NotImplementedError("Constraints not implemented")

        if comm is None:
            comm = PETSc.COMM_WORLD
        if hasattr(comm, "tompi4py"):
            comm = comm.tompi4py()

        tao_objective = TAOObjective(problem.reduced_functional)

        vec_interface = PETScVecInterface(
            tuple(control.control for control in tao_objective.reduced_functional.controls),
            comm=comm)
        n, N = vec_interface.n, vec_interface.N
        to_petsc, from_petsc = vec_interface.to_petsc, vec_interface.from_petsc

        tao = PETSc.TAO().create(comm=comm)

        def objective_gradient(tao, x, g):
            m = tao_objective.new_control_variable()
            from_petsc(x, m)
            J_val, dJ = tao_objective.objective_gradient(m)
            to_petsc(g, dJ)
            return J_val

        tao.setObjectiveGradient(objective_gradient, None)

        def hessian(tao, x, H, P):
            H.getPythonContext().set_control_variable(x)

        class Hessian:
            """:class:`petsc4py.PETSc.Mat` context.
            """

            def __init__(self):
                self._shift = 0.0

            @cached_property
            def _m(self):
                return tao_objective.new_control_variable()

            def set_control_variable(self, x):
                from_petsc(x, self._m)
                self._shift = 0.0

            def shift(self, A, alpha):
                self._shift += alpha

            def mult(self, A, x, y):
                m_dot = tao_objective.new_control_variable()
                from_petsc(x, m_dot)
                ddJ = tao_objective.hessian(self._m, m_dot)
                to_petsc(y, ddJ)
                if self._shift != 0.0:
                    y.axpy(self._shift, x)

        H_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                            Hessian(), comm=comm)
        H_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        H_matrix.setUp()
        tao.setHessian(hessian, H_matrix)

        class GradientNorm:
            """:class:`petsc4py.PETSc.Mat` context.
            """

            def mult(self, A, x, y):
                dJ = tao_objective.new_dual_control_variable()
                from_petsc(x, dJ)
                assert len(tao_objective.reduced_functional.controls) == len(dJ)
                dJ = tuple(control._ad_convert_riesz(dJ_i, riesz_map=control.riesz_map)
                           for control, dJ_i in zip(tao_objective.reduced_functional.controls, dJ))
                to_petsc(y, dJ)

        M_inv_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                GradientNorm(), comm=comm)
        M_inv_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        M_inv_matrix.setUp()
        tao.setGradientNorm(M_inv_matrix)

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

        self.options = OptionsManager(parameters, None)
        self.options.set_from_options(tao)

        if tao.getType() in {PETSc.TAO.Type.LMVM, PETSc.TAO.Type.BLMVM}:
            class InitialHessian:
                """:class:`petsc4py.PETSc.Mat` context.
                """

            class InitialHessianPreconditioner:
                """:class:`petsc4py.PETSc.PC` context.
                """

                def apply(self, pc, x, y):
                    dJ = tao_objective.new_dual_control_variable()
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
        with self.options.inserted_options():
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
        self._vec_interface.to_petsc(self.x, m)
        with self.options.inserted_options():
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
