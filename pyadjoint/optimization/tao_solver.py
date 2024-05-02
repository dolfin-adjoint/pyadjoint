from .optimization_solver import OptimizationSolver


try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None


class TAOSolver(OptimizationSolver):
    """Use TAO to solve an optimization problem.
    """

    def __init__(self, problem, parameters):
        super().__init__(problem, parameters)

    def solve(self):
        raise NotImplementedError
