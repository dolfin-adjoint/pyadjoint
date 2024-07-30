from enum import Enum
from functools import cached_property
import weakref

import numpy as np

from ..enlisting import Enlist
from .optimization_problem import MinimizationProblem
from .optimization_solver import OptimizationSolver


try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None

__all__ = \
    [
        "TAOSolver"
    ]


def attach_destroy_finalizer(obj, *args):
    def finalize_callback(*args):
        for arg in args:
            if arg is not None:
                arg.destroy()

    finalize = weakref.finalize(obj, finalize_callback,
                                *args)
    finalize.atexit = False


class PETScVecInterface:
    """Interface for conversion between :class:`OverloadedType` objects and
    :class:`petsc4py.PETSc.Vec` objects.

    This uses the generic interface provided by :class:`OverloadedType`.

    Args:
        X (OverloadedType or Sequence[OverloadedType]): One or more variables
            defining the size of data to be stored.
        comm (petsc4py.PETSc.Comm or mpi4py.MPI.Comm): Communicator.
    """

    def __init__(self, X, *, comm=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        X = Enlist(X)
        if comm is None:
            comm = PETSc.COMM_WORLD
        if hasattr(comm, "tompi4py"):
            comm = comm.tompi4py()

        vecs = tuple(x._ad_to_petsc() for x in X)
        n = sum(vec.getLocalSize() for vec in vecs)
        N = sum(vec.getSize() for vec in vecs)
        _, isets = PETSc.Vec().concatenate(vecs)
        for vec in vecs:
            vec.destroy()
        del vecs

        self._comm = comm
        self._n = n
        self._N = N
        self._isets = isets

        attach_destroy_finalizer(self, *self._isets)

    @property
    def comm(self):
        """Communicator.
        """

        return self._comm

    @property
    def n(self):
        """Total number of process local DoFs, summed over all variables."""

        return self._n

    @property
    def N(self):
        """Total number of global degrees of freedom, summed over all
        variables.
        """

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

    def from_petsc(self, y, X):
        """Copy data from a :class:`petsc4py.PETSc.Vec` to variables.

        Args:
            y (petsc4py.PETSc.Vec): The input :class:`petsc4py.PETSc.Vec`.
            X (OverloadedType or Sequence[OverloadedType]): The output
                variables.
        """

        X = Enlist(X)
        if len(X) != len(self._isets):
            raise ValueError("Invalid length")
        for iset, x in zip(self._isets, X):
            y_sub = y.getSubVector(iset)
            x._ad_from_petsc(y_sub)
            y.restoreSubVector(iset, y_sub)

    def to_petsc(self, x, Y):
        """Copy data from variables to a :class:`petsc4py.PETSc.Vec`.

        Args:
            x (petsc4py.PETSc.Vec): The output :class:`petsc4py.PETSc.Vec`.
            Y (OverloadedType or Sequence[OverloadedType]):
                Values for input variables.
        """

        Y = Enlist(Y)
        for iset, y in zip(self._isets, Y):
            x_sub = x.getSubVector(iset)
            y_vec = y._ad_to_petsc()
            y_vec.copy(result=x_sub)
            y_vec.destroy()
            x_sub.restoreSubVector(iset, x_sub)


class PETScOptions:
    def __init__(self, options_prefix):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        self._options_prefix = options_prefix
        self._options = PETSc.Options()
        self._keys = {}

        def finalize_callback(options_prefix, options, keys):
            for key in keys:
                key = f"{options_prefix:s}{key:s}"
                if key in options:
                    del options[key]

        finalize = weakref.finalize(
            self, finalize_callback,
            self._options_prefix, self._options, self._keys)
        finalize.atexit = False

    @property
    def options_prefix(self):
        return self._options_prefix

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return self._options[f"{self.options_prefix:s}{key:s}"]

    def __setitem__(self, key, value):
        self._keys[key] = None
        self._options[f"{self.options_prefix:s}{key:s}"] = value

    def __delitem__(self, key):
        del self._keys[key]
        del self._options[f"{self.options_prefix:s}{key:s}"]

    def clear(self):
        for key in tuple(self._keys):
            del self[key]

    def update(self, other):
        for key, value in other.items():
            self[key] = value


class BoundType(Enum):
    LOWER = 0
    UPPER = 1


class TAOObjective:
    def __init__(self, rf):
        self._reduced_functional = rf

    @property
    def reduced_functional(self):
        return self._reduced_functional

    def objective_gradient(self, M):
        M = Enlist(M)
        J = self.reduced_functional(tuple(m._ad_copy() for m in M))
        _ = self.reduced_functional.derivative()
        dJ = tuple(m.control.block_variable.adj_value
                   for m in self.reduced_functional.controls)
        return J, M.delist(dJ)

    def hessian(self, M, M_dot):
        M = Enlist(M)
        M_dot = Enlist(M_dot)
        _ = self.reduced_functional(tuple(m._ad_copy() for m in M))
        _ = self.reduced_functional.derivative()
        _ = self.reduced_functional.hessian(tuple(m_dot._ad_copy() for m_dot in M_dot))
        ddJ = tuple(m.control.block_variable.hessian_value
                    for m in self.reduced_functional.controls)
        return M.delist(ddJ)

    def new_M(self):
        # Not initialized to zero
        return tuple(m.control._ad_copy()
                     for m in self.reduced_functional.controls)

    def new_M_dual(self):
        # Not initialized to zero, requires adjoint or Hessian action values to
        # have already been computed
        M_dual = []
        for m in self.reduced_functional.controls:
            bv = m.control.block_variable
            if bv.adj_value is not None:
                M_dual.append(bv.adj_value)
            elif bv.hessian_value is not None:
                M_dual.append(bv.hessian_value)
            else:
                raise RuntimeError("Unable to instantiate new dual space "
                                   "object")
        return tuple(m_dual._ad_copy() for m_dual in M_dual)


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

    def __init__(self, problem, parameters, *, comm=None, convert_options=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        if not isinstance(problem, MinimizationProblem):
            raise TypeError("MinimizationProblem required")
        if problem.constraints is not None:
            raise NotImplementedError("Constraints not implemented")

        if comm is None:
            comm = PETSc.COMM_WORLD
        if hasattr(comm, "tompi4py"):
            comm = comm.tompi4py()
        if convert_options is None:
            convert_options = {}

        taoobjective = TAOObjective(problem.reduced_functional)

        vec_interface = PETScVecInterface(
            tuple(m.control for m in taoobjective.reduced_functional.controls),
            comm=comm)
        n, N = vec_interface.n, vec_interface.N
        to_petsc, from_petsc = vec_interface.to_petsc, vec_interface.from_petsc
        tao = PETSc.TAO().create(comm=comm)

        def objective_gradient(tao, x, g):
            M = taoobjective.new_M()
            from_petsc(x, M)
            J_val, dJ = taoobjective.objective_gradient(M)
            to_petsc(g, dJ)
            return J_val

        tao.setObjectiveGradient(objective_gradient, None)

        def hessian(tao, x, H, P):
            H.getPythonContext().set_M(x)

        class Hessian:
            def __init__(self):
                self._shift = 0.0

            @cached_property
            def _M(self):
                return taoobjective.new_M()

            def set_M(self, x):
                from_petsc(x, self._M)
                self._shift = 0.0

            def shift(self, A, alpha):
                self._shift += alpha

            def mult(self, A, x, y):
                M_dot = taoobjective.new_M()
                from_petsc(x, M_dot)
                ddJ = taoobjective.hessian(self._M, M_dot)
                to_petsc(y, ddJ)
                if self._shift != 0.0:
                    y.axpy(self._shift, x)

        H_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                            Hessian(), comm=comm)
        H_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        H_matrix.setUp()
        tao.setHessian(hessian, H_matrix)

        class GradientNorm:
            def mult(self, A, x, y):
                X = taoobjective.new_M_dual()
                from_petsc(x, X)
                assert len(taoobjective.reduced_functional.controls) == len(X)
                X = tuple(m._ad_convert_type(x, options=convert_options)
                          for m, x in zip(taoobjective.reduced_functional.controls, X))
                to_petsc(y, X)

        M_inv_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                GradientNorm(), comm=comm)
        M_inv_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        M_inv_matrix.setUp()
        tao.setGradientNorm(M_inv_matrix)

        if problem.bounds is not None:
            lbs = []
            ubs = []
            assert len(problem.bounds) == len(problem.reduced_functional.controls)
            for (lb, ub), control in zip(problem.bounds, taoobjective.reduced_functional.controls):
                lb = self._prepared_bound(
                    control, lb, bound_type=BoundType.LOWER)
                ub = self._prepared_bound(
                    control, ub, bound_type=BoundType.UPPER)
                lbs.append(lb)
                ubs.append(ub)

            lb_vec = vec_interface.new_petsc()
            ub_vec = vec_interface.new_petsc()
            to_petsc(lb_vec, lbs)
            to_petsc(ub_vec, ubs)
            tao.setVariableBounds(lb_vec, ub_vec)
            lb_vec.destroy()
            ub_vec.destroy()

        options = PETScOptions(f"_pyadjoint__{tao.name:s}_")
        options.update(parameters)
        tao.setOptionsPrefix(options.options_prefix)

        tao.setFromOptions()

        if tao.getType() in {PETSc.TAO.Type.LMVM, PETSc.TAO.Type.BLMVM}:
            class InitialHessian:
                pass

            class InitialHessianPreconditioner:
                def apply(self, pc, x, y):
                    X = taoobjective.new_M_dual()
                    from_petsc(x, X)
                    assert len(taoobjective.reduced_functional.controls) == len(X)
                    X = tuple(m._ad_convert_type(x, options=convert_options)
                              for m, x in zip(taoobjective.reduced_functional.controls, X))
                    to_petsc(y, X)

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
        self._taoobjective = taoobjective
        self._vec_interface = vec_interface
        self._tao = tao
        self._x = x

        attach_destroy_finalizer(
            self, tao, H_matrix, M_inv_matrix, B_0_matrix_pc, B_0_matrix, x)

    @property
    def taoobjective(self):
        return self._taoobjective

    @property
    def tao(self):
        return self._tao

    @property
    def x(self):
        return self._x

    @staticmethod
    def _prepared_bound(control, bound, bound_type):
        if bound is None:
            bound = {BoundType.LOWER: np.finfo(PETSc.ScalarType).min,
                     BoundType.UPPER: np.finfo(PETSc.ScalarType).max}[bound_type]
        bound_val = bound
        bound = control.control._ad_copy()
        bound._ad_assign(bound_val)
        return bound

    def solve(self):
        """Solve the optimisation problem.

        Returns:
            OverloadedType or tuple[OverloadedType]: The solution.
        """
        M = tuple(
            m.tape_value()._ad_copy()
            for m in self.taoobjective.reduced_functional.controls
        )
        self._vec_interface.to_petsc(self.x, M)
        self.tao.solve()
        self._vec_interface.from_petsc(self.x, M)
        return self.taoobjective.reduced_functional.controls.delist(M)
