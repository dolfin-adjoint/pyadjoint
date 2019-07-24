import backend
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from .solving import SolveBlock


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

    def solve(self, **kwargs):
        """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation)."""

        annotate = annotate_tape(kwargs)
        if annotate:
            tape = get_working_tape()
            problem = self._ad_problem
            sb_kwargs = SolveBlock.pop_kwargs(kwargs)
            sb_kwargs.update(kwargs)
            block = NonlinearVariationalSolveBlock(problem._ad_F == 0,
                                                   problem._ad_u,
                                                   problem._ad_bcs,
                                                   *self._ad_args,
                                                   problem_J=problem._ad_J,
                                                   solver_params=self.parameters,
                                                   solver_kwargs=self._ad_kwargs,
                                                   **sb_kwargs)
            tape.add_block(block)

        with stop_annotating():
            out = super(NonlinearVariationalSolver, self).solve()

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

    def solve(self, **kwargs):
        """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation)."""

        annotate = annotate_tape(kwargs)
        if annotate:
            tape = get_working_tape()
            problem = self._ad_problem
            sb_kwargs = SolveBlock.pop_kwargs(kwargs)
            sb_kwargs.update(kwargs)
            block = LinearVariationalSolveBlock(problem._ad_a == problem._ad_L,
                                                problem._ad_u,
                                                problem._ad_bcs,
                                                *self._ad_args,
                                                solver_params=self.parameters,
                                                solver_kwargs=self._ad_kwargs,
                                                **sb_kwargs)
            tape.add_block(block)

        with stop_annotating():
            out = super(LinearVariationalSolver, self).solve()

        if annotate:
            block.add_output(self._ad_problem._ad_u.create_block_variable())

        return out


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
        problem = backend.NonlinearVariationalProblem(lhs, func, bcs, J=J)
        solver = backend.NonlinearVariationalSolver(problem, **self.nonlin_solver_kwargs)
        solver.parameters.update(self.nonlin_solver_params)
        solver.solve()
        return func


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
