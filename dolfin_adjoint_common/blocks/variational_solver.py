from . import SolveBlock


class NonlinearVariationalSolveBlock(SolveBlock):
    def __init__(self, *args, **kwargs):
        self.nonlin_problem_J = kwargs.pop("problem_J")
        self.nonlin_solver_params = kwargs.pop("solver_params").copy()
        self.nonlin_solver_kwargs = kwargs.pop("solver_kwargs")

        kwargs.update(self.nonlin_solver_kwargs)
        super(NonlinearVariationalSolveBlock, self).__init__(*args, **kwargs)

        if self.nonlin_problem_J is not None:
            for coeff in self.nonlin_problem_J.coefficients():
                self.add_dependency(coeff, no_duplicates=True)

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        J = self.nonlin_problem_J
        if J is not None:
            J = self._replace_form(J, func)
        problem = self.backend.NonlinearVariationalProblem(lhs, func, bcs, J=J)
        solver = self.backend.NonlinearVariationalSolver(problem, **self.nonlin_solver_kwargs)
        solver.parameters.update(self.nonlin_solver_params)
        solver.solve()
        return func

