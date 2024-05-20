from functools import cached_property
import weakref

from ..enlisting import Enlist
from .optimization_problem import MinimizationProblem
from .optimization_solver import OptimizationSolver


import numpy as np
try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None

__all__ = \
    [
        "TAOSolver"
    ]


class PETScVecInterface:
    def __init__(self, X, *, comm=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        X = Enlist(X)
        if comm is None:
            comm = PETSc.COMM_WORLD
        if hasattr(comm, "tompi4py"):
            comm = comm.tompi4py()

        indices = []
        n = 0
        N = 0
        for x in X:
            y = x._ad_copy()
            # Global vector
            y_a = np.zeros(y._ad_dim(), dtype=PETSc.ScalarType)
            _, x_n = y._ad_assign_numpy(y, y_a, offset=0)
            del y, y_a
            indices.append((n, n + x_n))
            n += x_n
            N += x._ad_dim()

        self._comm = comm
        self._indices = tuple(indices)
        self._n = n
        self._N = N

    @property
    def comm(self):
        return self._comm

    @property
    def indices(self):
        return self._indices

    @property
    def n(self):
        return self._n

    @property
    def N(self):
        return self._N

    def new_petsc(self):
        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes((self.n, self.N))
        vec.setUp()
        return vec

    def from_petsc(self, y, X):
        X = Enlist(X)
        y_a = y.getArray(True)

        if y_a.shape != (self.n,):
            raise ValueError("Invalid shape")
        if len(X) != len(self.indices):
            raise ValueError("Invalid length")

        for (i0, i1), x in zip(self.indices, X):
            _, x_i1 = x._ad_assign_numpy(x, y_a, offset=i0)
            if i1 != x_i1:
                raise ValueError("Invalid index")

    def to_petsc(self, x, Y):
        Y = Enlist(Y)
        if len(Y) != len(self.indices):
            raise ValueError("Invalid length")

        x_a = np.zeros(self.n, dtype=PETSc.ScalarType)
        for (i0, i1), y in zip(self.indices, Y):
            x_a[i0:i1] = y._ad_to_list(y)
        x.setArray(x_a)


class PETScOptions:
    def __init__(self, options_prefix):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        self.options_prefix = options_prefix
        self._options = PETSc.Options()
        self._keys = {}  # Use dict as an ordered set

        def finalize_callback(options_prefix, options, keys):
            for key in keys:
                key = f"{options_prefix:s}{key:s}"
                if key in options:
                    del options[key]

        finalize = weakref.finalize(
            self, finalize_callback,
            self.options_prefix, self._options, self._keys)
        finalize.atexit = False

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

    def update(self, d):
        for key, value in d.items():
            self[key] = value


class TAOObjective:
    def __init__(self, rf):
        self.reduced_functional = rf

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
        comm (petsc4py.PETSc.Comm or mpi4py.MPI.Comm): Communicator.
        inner_product (str): Defines the Riesz map. Used to define the
            inner product for derivatives with respect to the control.
    """

    def __init__(self, problem, parameters, *, comm=None, inner_product="L2"):
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
                dJ = taoobjective.new_M_dual()
                from_petsc(x, dJ)
                assert len(taoobjective.reduced_functional.controls) == len(dJ)
                dJ = tuple(m._ad_convert_type(dJ_, {"riesz_representation": inner_product})
                           for m, dJ_ in zip(taoobjective.reduced_functional.controls, dJ))
                to_petsc(y, dJ)

        M_inv_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                GradientNorm(), comm=comm)
        M_inv_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        M_inv_matrix.setUp()
        tao.setGradientNorm(M_inv_matrix)

        if problem.bounds is not None:
            def convert_bound(m, b, default):
                if b is None:
                    b = default
                return m._ad_convert_type(b)

            x_lb = vec_interface.new_petsc()
            x_ub = vec_interface.new_petsc()
            assert len(taoobjective.reduced_functional.controls) == len(problem.bounds)
            to_petsc(x_lb, tuple(convert_bound(m, lb, np.finfo(PETSc.ScalarType).min)
                                 for m, (lb, _) in zip(taoobjective.reduced_functional.controls, problem.bounds)))
            to_petsc(x_ub, tuple(convert_bound(m, ub, np.finfo(PETSc.ScalarType).max)
                                 for m, (_, ub) in zip(taoobjective.reduced_functional.controls, problem.bounds)))
            tao.setVariableBounds(x_lb, x_ub)
            x_lb.destroy()
            x_ub.destroy()

        options = PETScOptions(f"_pyadjoint__{tao.name:s}_")
        options.update(parameters)
        tao.setOptionsPrefix(options.options_prefix)

        tao.setFromOptions()

        super().__init__(problem, parameters)
        self.taoobjective = taoobjective
        self.vec_interface = vec_interface
        self.tao = tao

        def finalize_callback(tao):
            tao.destroy()

        finalize = weakref.finalize(self, finalize_callback,
                                    tao)
        finalize.atexit = False

    def solve(self):
        """Solve the optimisation problem.

        Returns:
            OverloadedType or tuple[OverloadedType]: The solution.
        """

        M = tuple(m.tape_value()._ad_copy()
                  for m in self.taoobjective.reduced_functional.controls)
        x = self.vec_interface.new_petsc()
        self.vec_interface.to_petsc(x, M)
        self.tao.solve(x)
        self.vec_interface.from_petsc(x, M)
        x.destroy()
        return self.taoobjective.reduced_functional.controls.delist(M)
