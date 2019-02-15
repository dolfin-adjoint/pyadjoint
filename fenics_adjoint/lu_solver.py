import backend
from pyadjoint.tape import annotate_tape, get_working_tape
from .types import compat
from .solving import SolveBlock


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
            sb_kwargs = SolveBlock.pop_kwargs(kwargs)
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


class LUSolveBlockHelper(object):
    def __init__(self):
        self.forward_solver = None
        self.adjoint_solver = None

    def reset(self):
        self.forward_solver = None
        self.adjoint_solver = None


class LUSolveBlock(SolveBlock):
    def __init__(self, *args, **kwargs):
        super(LUSolveBlock, self).__init__(*args, **kwargs)
        self.lu_solver_parameters = kwargs.pop("lu_solver_parameters")
        self.block_helper = kwargs.pop("block_helper")
        self.method = kwargs.pop("lu_solver_method")

    def _assemble_and_solve_adj_eq(self, dFdu_form, dJdu):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()

        solver = self.block_helper.adjoint_solver
        if solver is None:
            if self.assemble_system:
                rhs_bcs_form = backend.inner(backend.Function(self.function_space),
                                             dFdu_form.arguments()[0]) * backend.dx
                A, _ = backend.assemble_system(dFdu_form, rhs_bcs_form, bcs)
            else:
                A = compat.assemble_adjoint_value(dFdu_form)
                [bc.apply(A) for bc in bcs]

            solver = backend.LUSolver(A, self.method)
            self.block_helper.adjoint_solver = solver

        solver.parameters.update(self.lu_solver_parameters)
        [bc.apply(dJdu) for bc in bcs]

        adj_sol = backend.Function(self.function_space)
        solver.solve(adj_sol.vector(), dJdu)

        adj_sol_bdy = compat.function_from_vector(self.function_space, dJdu_copy - compat.assemble_adjoint_value(
            backend.action(dFdu_form, adj_sol)))

        return adj_sol, adj_sol_bdy

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        solver = self.block_helper.forward_solver
        if solver is None:
            if self.assemble_system:
                A, _ = backend.assemble_system(lhs, rhs, bcs)
            else:
                A = compat.assemble_adjoint_value(lhs)
                [bc.apply(A) for bc in bcs]

            solver = backend.LUSolver(A, self.method)
            self.block_helper.forward_solver = solver

        if self.assemble_system:
            system_assembler = backend.SystemAssembler(lhs, rhs, bcs)
            b = backend.Function(self.function_space).vector()
            system_assembler.assemble(b)
        else:
            b = compat.assemble_adjoint_value(rhs)
            [bc.apply(b) for bc in bcs]

        solver.parameters.update(self.lu_solver_parameters)
        solver.solve(func.vector(), b)
        return func
