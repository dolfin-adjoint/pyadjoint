import backend
from . import SolveVarFormBlock


class LinearVariationalSolveBlock(SolveVarFormBlock):
    def __init__(self, equation, func, bcs,
                 problem_args, problem_kwargs,
                 solver_params, solver_args,
                 solver_kwargs, solve_args, solve_kwargs,
                 **kwargs):
        func = func
        bcs = bcs

        self.problem_args = problem_args
        self.problem_kwargs = problem_kwargs
        self.solver_params = solver_params.copy()
        self.solver_args = solver_args
        self.solver_kwargs = solver_kwargs
        self.solve_args = solve_args
        self.solve_kwargs = solve_kwargs

        super(LinearVariationalSolveBlock, self).__init__(equation, func, bcs, **kwargs)

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


class NonlinearVariationalSolveBlock(SolveVarFormBlock):
    def __init__(self, equation, func, bcs, problem_J,
                 problem_args, problem_kwargs,
                 solver_params, solver_args,
                 solver_kwargs, solve_args, solve_kwargs,
                 **kwargs):
        func = func
        bcs = bcs

        self.problem_J = problem_J
        self.problem_args = problem_args
        self.problem_kwargs = problem_kwargs
        self.solver_params = solver_params.copy()
        self.solver_args = solver_args
        self.solver_kwargs = solver_kwargs
        self.solve_args = solve_args
        self.solve_kwargs = solve_kwargs

        super(NonlinearVariationalSolveBlock, self).__init__(equation, func, bcs, **kwargs)

        if self.problem_J is not None:
            for coeff in self.problem_J.coefficients():
                self.add_dependency(coeff, no_duplicates=True)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        if len(self.adj_args) <= 0 and len(self.adj_kwargs) <= 0:
            params_key = "{}_solver".format(self.solver_params["nonlinear_solver"])

            if params_key in self.solver_params:
                method = self.solver_params[params_key]["linear_solver"]
                precond = self.solver_params[params_key]["preconditioner"]
                self.adj_args = [method, precond]

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        J = self.problem_J
        if J is not None:
            J = self._replace_form(J, func)
        problem = self.backend.NonlinearVariationalProblem(lhs, func, bcs, J=J,
                                                           *self.problem_args, **self.problem_kwargs)
        solver = self.backend.NonlinearVariationalSolver(problem, *self.solver_args, **self.solver_kwargs)
        solver.parameters.update(self.solver_params)
        solver.solve(*self.solve_args, **self.solve_kwargs)
        return func
