import backend
from . import GenericSolveBlock


class LinearVariationalSolveBlock(GenericSolveBlock):
    def __init__(self, equation, func, bcs,
                 problem_args, problem_kwargs,
                 solver_params, solver_args,
                 solver_kwargs, solve_args, solve_kwargs,
                 **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs
        func = func
        bcs = bcs

        self.problem_args = problem_args
        self.problem_kwargs = problem_kwargs
        self.solver_params = solver_params.copy()
        self.solver_args = solver_args
        self.solver_kwargs = solver_kwargs
        self.solve_args = solve_args
        self.solve_kwargs = solve_kwargs

        super(LinearVariationalSolveBlock, self).__init__(lhs, rhs, func, bcs, **kwargs)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        if len(self.adj_args) <= 0 and len(self.adj_kwargs) <= 0:
            method = self.solver_params["linear_solver"]
            precond = self.solver_params["preconditioner"]
            self.adj_args = [method, precond]

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        problem = backend.LinearVariationalProblem(lhs, rhs, func, bcs, *self.problem_args, **self.problem_kwargs)
        solver = backend.LinearVariationalSolver(problem, *self.solver_args, **self.solver_kwargs)
        solver.parameters.update(self.solver_params)
        solver.solve(*self.solve_args, **self.solve_kwargs)
        return func
