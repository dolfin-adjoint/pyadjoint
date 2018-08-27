import backend
from pyadjoint.tape import annotate_tape, get_working_tape
from .solving import SolveBlock


class LUSolver(backend.LUSolver):
    '''This object is overloaded so that solves using this class are automatically annotated,
    so that libadjoint can automatically derive the adjoint and tangent linear models.'''
    def __init__(self, *args):
        self.__global_list_idx__ = None

        if len(args) > 0:
            self.matrix = args[0]

        backend.LUSolver.__init__(self, *args)

    def set_operator(self, operator):
        self.matrix = operator
        backend.LUSolver.set_operator(self, operator)

    def solve(self, *args, **kwargs):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation).'''

        annotate = annotate_tape(kwargs)

        if annotate:
            if len(args) == 2:
                try:
                    A = self.matrix.form
                except AttributeError:
                    raise Exception("Your matrix A has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

                try:
                    self.op_bcs = self.matrix.bcs
                except AttributeError:
                    self.op_bcs = []

                try:
                    x = args[0].function
                except AttributeError:
                    raise Exception("Your solution x has to have a .function attribute; is it the .vector() of a Function?")

                try:
                    b = args[1].form
                except AttributeError:
                    raise Exception("Your RHS b has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

                try:
                    eq_bcs = self.op_bcs + args[1].bcs
                except AttributeError:
                    eq_bcs = self.op_bcs

            elif len(args) == 3:
                A = args[0].form
                try:
                    x = args[1].function
                except AttributeError:
                    raise Exception("Your solution x has to have a .function attribute; is it the .vector() of a Function?")

                try:
                    self.op_bcs = A.bcs
                except AttributeError:
                    self.op_bcs = []

                try:
                    b = args[2].form
                except AttributeError:
                    raise Exception("Your RHS b has to have the .form attribute: was it assembled after from dolfin_adjoint import *?")

                try:
                    eq_bcs = self.op_bcs + args[2].bcs
                except AttributeError:
                    eq_bcs = self.op_bcs

            else:
                raise Exception("LUSolver.solve() must be called with either (A, x, b) or (x, b).")

            tape = get_working_tape()
            # TODO need to extend annotation to remember more about solvers.
            block = SolveBlock(A == b,
                               x,
                               eq_bcs,
                               solver_parameters={"linear_solver": "lu"})
            tape.add_block(block)

        out = backend.LUSolver.solve(self, *args, **kwargs)

        if annotate:
            block.add_output(x.create_block_variable())

        return out
