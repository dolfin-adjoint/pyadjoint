import backend
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape
from .solving import SolveBlock


class NonlinearVariationalProblem(backend.NonlinearVariationalProblem):

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

    def __init__(self, problem, *args, **kwargs):
        super(NonlinearVariationalSolver, self).__init__(problem, *args, **kwargs)
        self._ad_problem = problem
        self._ad_args = args
        self._ad_kwargs = kwargs

    def solve(self, **kwargs):
        annotate = annotate_tape(kwargs)
        if annotate:
            tape = get_working_tape()
            problem = self._ad_problem
            # TODO need to extend annotation to remember more about solvers.
            block = SolveBlock(problem._ad_F == 0,
                               problem._ad_u,
                               problem._ad_bcs,
                               *self._ad_args,
                               **self._ad_kwargs)
            tape.add_block(block)

        with stop_annotating():
            out = super(NonlinearVariationalSolver, self).solve()

        if annotate:
            block.add_output(self._ad_problem._ad_u.create_block_output())

        return out


class LinearVariationalProblem(backend.LinearVariationalProblem):

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

    def __init__(self, problem, *args, **kwargs):
        super(LinearVariationalSolver, self).__init__(problem, *args, **kwargs)
        self._ad_problem = problem
        self._ad_args = args
        self._ad_kwargs = kwargs

    def solve(self, **kwargs):
        annotate = annotate_tape(kwargs)
        if annotate:
            tape = get_working_tape()
            problem = self._ad_problem
            block = SolveBlock(problem._ad_a == problem._ad_L,
                               problem._ad_u,
                               problem._ad_bcs,
                               *self._ad_args,
                               **self._ad_kwargs)
            tape.add_block(block)

        with stop_annotating():
            out = super(LinearVariationalSolver, self).solve()

        if annotate:
            block.add_output(self._ad_problem._ad_u.create_block_output())

        return out
