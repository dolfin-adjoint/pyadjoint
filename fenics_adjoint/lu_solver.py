import backend
from pyadjoint.tape import annotate_tape, get_working_tape
from dolfin_adjoint_common import compat

from .blocks import LUSolveBlock, LUSolveBlockHelper

compat = compat.compat(backend)


class LUSolver(backend.LUSolver):
    def __init__(self, *args, **kwargs):
        backend.LUSolver.__init__(self, *args, **kwargs)

        A = kwargs.pop("A", None)
        method = kwargs.pop("method", "default")

        next_arg_idx = 0
        if len(args) > 0 and isinstance(args[0], compat.MatrixType):
            A = args[0]
            next_arg_idx = 1
        elif len(args) > 1 and isinstance(args[1], compat.MatrixType):
            A = args[1]
            next_arg_idx = 2

        if len(args) > next_arg_idx and isinstance(args[next_arg_idx], str):
            method = args[next_arg_idx]

        self.operator = A
        self.method = method
        self.solver_parameters = {}
        self.block_helper = LUSolveBlockHelper()

    def set_operator(self, arg0):
        self.operator = arg0
        self.block_helper = LUSolveBlockHelper()
        return backend.LUSolver.set_operator(self, arg0)

    def solve(self, *args, **kwargs):
        annotate = annotate_tape(kwargs)

        if annotate:
            if len(args) == 3:
                block_helper = LUSolveBlockHelper()
                A = args[0]
                x = args[1]
                b = args[2]
            elif len(args) == 2:
                block_helper = self.block_helper
                A = self.operator
                x = args[0]
                b = args[1]

            u = x.function
            parameters = self.parameters.copy()

            tape = get_working_tape()
            sb_kwargs = LUSolveBlock.pop_kwargs(kwargs)
            block = LUSolveBlock(A, x, b,
                                 lu_solver_parameters=parameters,
                                 block_helper=block_helper,
                                 lu_solver_method=self.method,
                                 **sb_kwargs)
            tape.add_block(block)

        out = backend.LUSolver.solve(self, *args, **kwargs)

        if annotate:
            block.add_output(u.create_block_variable())

        return out
