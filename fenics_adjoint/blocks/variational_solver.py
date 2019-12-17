

class LinearVariationalSolveBlock(SolveBlock):
    def __init__(self, *args, **kwargs):
        self.lin_solver_params = kwargs.pop("solver_params").copy()
        self.lin_solver_kwargs = kwargs.pop("solver_kwargs")

        kwargs.update(self.lin_solver_kwargs)
        super(LinearVariationalSolveBlock, self).__init__(*args, **kwargs)

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        problem = backend.LinearVariationalProblem(lhs, rhs, func, bcs)
        solver = backend.LinearVariationalSolver(problem, **self.lin_solver_kwargs)
        solver.parameters.update(self.lin_solver_params)
        solver.solve()
        return func
