from ..enlisting import Enlist
from .optimization_solver import OptimizationSolver


import numpy as np
try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None


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
        for x in X:
            indices.append((n, n + x._ad_dim()))
            n += x._ad_dim()
        if comm.size > 0:
            import mpi4py.MPI as MPI
            N = comm.allreduce(n, op=MPI.SUM)

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
            if x._ad_dim() != i1 - i0:
                raise ValueError("Invalid length")

        for (i0, _), x in zip(self.indices, X):
            x._ad_assign_numpy(x, y_a, offset=i0)

    def to_petsc(self, x, Y):
        Y = Enlist(Y)
        if len(Y) != len(self.indices):
            raise ValueError("Invalid length")
        for (i0, i1), y in zip(self.indices, Y):
            if y._ad_dim() != i1 - i0:
                raise ValueError("Invalid length")

        x_a = np.zeros(self.n, dtype=PETSc.ScalarType)
        for (i0, i1), y in zip(self.indices, Y):
            x_a[i0:i1] = y._ad_to_list(y)
        x.setArray(x_a)


class TAOSolver(OptimizationSolver):
    """Use TAO to solve an optimization problem.
    """

    def __init__(self, problem, parameters):
        super().__init__(problem, parameters)

    def solve(self):
        raise NotImplementedError
