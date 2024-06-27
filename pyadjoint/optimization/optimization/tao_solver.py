from functools import cached_property
import numbers
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
    """Interface for conversion between :class:`OverloadedType` objects and
    :class:`petsc4py.PETSc.Vec` objects.

    This uses the generic interface provided by :class:`OverloadedType`.
    Currently this gathers values on all processes.

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

        indices = []
        n = 0
        N = 0
        for x in X:
            x_n = x._ad_local_size()
            indices.append((n, n + x_n))
            n += x_n
            N += x._ad_dim()

        self._comm = comm
        self._indices = tuple(indices)
        self._n = n
        self._N = N
        self._index_concatenated_vecs = None

    @property
    def comm(self):
        """Communicator.
        """

        return self._comm

    @property
    def indices(self):
        """Local index ranges for variables.
        """

        return self._indices

    @property
    def n(self):
        """Total number of process local degrees of freedom, summed over all
        variables.
        """

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
        y_a = y.getArray(True)

        if y_a.shape != (self.n,):
            raise ValueError("Invalid shape")
        if len(X) != len(self._index_concatenated_vecs):
            raise ValueError("Invalid length")
        for i, x in enumerate(X):
            x._ad_petsc_vec().setArray(y_a[self._index_concatenated_vecs[i]])

    def to_petsc(self, x, Y):
        """Copy data from variables to a :class:`petsc4py.PETSc.Vec`.

        Args:
            x (petsc4py.PETSc.Vec): The output :class:`petsc4py.PETSc.Vec`.
            Y (numbers.Complex, OverloadedType or Sequence[OverloadedType]):
                Values for input variables.
        """

        Y = Enlist(Y)
        if len(Y) != len(self.indices):
            raise ValueError("Invalid length")
        x.setArray(
            self.concatenate_vecs([y._ad_petsc_vec() for y in Y]).getArray()
        )

    def concatenate_vecs(self, vecs):
        """Concatenate Vecs into a single Vec.

        Args:
            Y (`petsc4py.PETSc.Vec` or sequence of `petsc4py.PETSc.Vec`):
            The input Vecs to concatenate.
        """
        if not all(isinstance(vec, PETSc.Vec) for vec in vecs):
            raise ValueError("vecs should be a sequence of PETSc.Vec objects")

        vecs = Enlist(vecs)
        concatenated_vec, set_indexes = PETSc.Vec().concatenate(vecs)
        self._index_concatenated_vecs = [set_indexes[i].array for i in range(len(vecs))]
        return concatenated_vec


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
            if not all(len(bound) == 2 for bound in problem.bounds):
                raise ValueError("Each bound should be a tuple of length 2 (lb, ub)")
            if not len(problem.bounds) == len(problem.reduced_functional.controls):
                raise ValueError(
                    "bounds should be of length number of controls of the ReducedFunctional"
                )

            array_size = problem.reduced_functional.controls[0]._ad_local_size()
            concatenate_vecs = vec_interface.concatenate_vecs
            x_lb = concatenate_vecs(
                tuple(
                    PETSc.Vec().createWithArray(
                        np.full((array_size,), np.finfo(PETSc.ScalarType).max,
                                dtype=PETSc.ScalarType)
                    ) if lb is None else lb._ad_petsc_vec()
                    for lb, _ in problem.bounds
                )
            )
            x_ub = concatenate_vecs(
                tuple(
                    PETSc.Vec().createWithArray(
                        np.full((array_size,), np.finfo(PETSc.ScalarType).max,
                                dtype=PETSc.ScalarType)
                    ) if ub is None else ub._ad_petsc_vec()
                    for _, ub in problem.bounds
                )
            )

            tao.setVariableBounds(x_lb, x_ub)
            x_lb.destroy()
            x_ub.destroy()

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
        self.taoobjective = taoobjective
        self.vec_interface = vec_interface
        self.tao = tao
        self.x = x

        def finalize_callback(*args):
            for arg in args:
                if arg is not None:
                    arg.destroy()

        finalize = weakref.finalize(
            self, finalize_callback,
            tao, H_matrix, M_inv_matrix, B_0_matrix_pc, B_0_matrix, x)
        finalize.atexit = False

    def solve(self):
        """Solve the optimisation problem.

        Returns:
            OverloadedType or tuple[OverloadedType]: The solution.
        """
        M = tuple(
            m.tape_value()._ad_copy()
            for m in self.taoobjective.reduced_functional.controls
        )
        self.vec_interface.to_petsc(self.x, M)
        self.tao.solve()
        self.vec_interface.from_petsc(self.x, M)
        return self.taoobjective.reduced_functional.controls.delist(M)
