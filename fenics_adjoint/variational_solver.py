import backend
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from .solving import SolveBlock


class NonlinearVariationalProblem(backend.NonlinearVariationalProblem):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that pyadjoint can automatically derive the adjoint and tangent linear models.'''
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
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that pyadjoint can automatically derive the adjoint and tangent linear models.'''
    @no_annotations
    def __init__(self, problem, *args, **kwargs):
        super(NonlinearVariationalSolver, self).__init__(problem, *args, **kwargs)
        self._ad_problem = problem
        self._ad_args = args
        self._ad_kwargs = kwargs

    def solve(self, **kwargs):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation).'''

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
            block.add_output(self._ad_problem._ad_u.create_block_variable())

        return out


class LinearVariationalProblem(backend.LinearVariationalProblem):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that pyadjoint can automatically derive the adjoint and tangent linear models.'''
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
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that pyadjoint can automatically derive the adjoint and tangent linear models.'''
    @no_annotations
    def __init__(self, problem, *args, **kwargs):
        super(LinearVariationalSolver, self).__init__(problem, *args, **kwargs)
        self._ad_problem = problem
        self._ad_args = args
        self._ad_kwargs = kwargs

    def solve(self, **kwargs):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation).'''

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
            block.add_output(self._ad_problem._ad_u.create_block_variable())

        return out
