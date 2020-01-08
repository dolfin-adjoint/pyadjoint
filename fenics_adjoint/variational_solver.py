import backend
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from .solving import GenericSolveBlock


class NonlinearVariationalProblem(backend.NonlinearVariationalProblem):
    """This object is overloaded so that solves using this class are automatically annotated,
    so that pyadjoint can automatically derive the adjoint and tangent linear models."""

    @no_annotations
    def __init__(self, F, u, bcs=None, J=None, *args, **kwargs):
        super(NonlinearVariationalProblem, self).__init__(F, u, bcs, J,
                                                          *args, **kwargs)
        self._ad_F = F
        self._ad_u = u
        self._ad_bcs = bcs
        self._ad_J = J
        self._ad_args = args
        self._ad_kwargs = kwargs


class NonlinearVariationalSolver(backend.NonlinearVariationalSolver):
    """This object is overloaded so that solves using this class are automatically annotated,
    so that pyadjoint can automatically derive the adjoint and tangent linear models."""

    @no_annotations
    def __init__(self, problem, *args, **kwargs):
        super(NonlinearVariationalSolver, self).__init__(problem, *args, **kwargs)
        self._ad_problem = problem
        self._ad_args = args
        self._ad_kwargs = kwargs

    def solve(self, *args, **kwargs):
        """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation)."""

        annotate = annotate_tape(kwargs)
        if annotate:
            tape = get_working_tape()
            problem = self._ad_problem
            sb_kwargs = NonlinearVariationalSolveBlock.pop_kwargs(kwargs)
            sb_kwargs.update(kwargs)
            block = NonlinearVariationalSolveBlock(problem._ad_F == 0,
                                                   problem._ad_u,
                                                   problem._ad_bcs,
                                                   problem_J=problem._ad_J,
                                                   problem_args=problem._ad_args,
                                                   problem_kwargs=problem._ad_kwargs,
                                                   solver_params=self.parameters,
                                                   solver_args=self._ad_args,
                                                   solver_kwargs=self._ad_kwargs,
                                                   solve_args=args,
                                                   solve_kwargs=kwargs,
                                                   **sb_kwargs)
            tape.add_block(block)

        with stop_annotating():
            out = super(NonlinearVariationalSolver, self).solve(*args, **kwargs)

        if annotate:
            block.add_output(self._ad_problem._ad_u.create_block_variable())

        return out


class LinearVariationalProblem(backend.LinearVariationalProblem):
    """This object is overloaded so that solves using this class are automatically annotated,
    so that pyadjoint can automatically derive the adjoint and tangent linear models."""

    @no_annotations
    def __init__(self, a, L, u, bcs=None, *args, **kwargs):
        super(LinearVariationalProblem, self).__init__(a, L, u, bcs,
                                                       *args, **kwargs)
        self._ad_a = a
        self._ad_L = L
        self._ad_u = u
        self._ad_bcs = bcs
        self._ad_args = args
        self._ad_kwargs = kwargs


class LinearVariationalSolver(backend.LinearVariationalSolver):
    """This object is overloaded so that solves using this class are automatically annotated,
    so that pyadjoint can automatically derive the adjoint and tangent linear models."""

    @no_annotations
    def __init__(self, problem, *args, **kwargs):
        super(LinearVariationalSolver, self).__init__(problem, *args, **kwargs)
        self._ad_problem = problem
        self._ad_args = args
        self._ad_kwargs = kwargs

    def solve(self, *args, **kwargs):
        """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation)."""

        annotate = annotate_tape(kwargs)
        if annotate:
            tape = get_working_tape()
            problem = self._ad_problem
            sb_kwargs = LinearVariationalSolveBlock.pop_kwargs(kwargs)
            sb_kwargs.update(kwargs)
            block = LinearVariationalSolveBlock(problem._ad_a == problem._ad_L,
                                                problem._ad_u,
                                                problem._ad_bcs,
                                                problem_args=problem._ad_args,
                                                problem_kwargs=problem._ad_kwargs,
                                                solver_params=self.parameters,
                                                solver_args=self._ad_args,
                                                solver_kwargs=self._ad_kwargs,
                                                solve_args=args,
                                                solve_kwargs=kwargs,
                                                **sb_kwargs)
            tape.add_block(block)

        with stop_annotating():
            out = super(LinearVariationalSolver, self).solve(*args, **kwargs)

        if annotate:
            block.add_output(self._ad_problem._ad_u.create_block_variable())

        return out


class NonlinearVariationalSolveBlock(GenericSolveBlock):
    def __init__(self, equation, func, bcs, problem_J,
                 problem_args, problem_kwargs,
                 solver_params, solver_args,
                 solver_kwargs, solve_args, solve_kwargs,
                 **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs
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

        super(NonlinearVariationalSolveBlock, self).__init__(lhs, rhs, func, bcs, **kwargs)

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
        problem = backend.NonlinearVariationalProblem(lhs, func, bcs, J=J, *self.problem_args, **self.problem_kwargs)
        solver = backend.NonlinearVariationalSolver(problem, *self.solver_args, **self.solver_kwargs)
        solver.parameters.update(self.solver_params)
        solver.solve(*self.solve_args, **self.solve_kwargs)
        return func


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
