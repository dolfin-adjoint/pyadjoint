import fenics as backend
from . import SolveLinearSystemBlock
from dolfin_adjoint_common import compat
compat = compat.compat(backend)


class LUSolveBlockHelper(object):
    def __init__(self):
        self.forward_solver = None
        self.adjoint_solver = None

    def reset(self):
        self.forward_solver = None
        self.adjoint_solver = None


class LUSolveBlock(SolveLinearSystemBlock):
    def __init__(self, A, u, b, *args, **kwargs):
        super(LUSolveBlock, self).__init__(A, u, b, **kwargs)
        self.lu_solver_parameters = kwargs.pop("lu_solver_parameters")
        self.block_helper = kwargs.pop("block_helper")
        self.method = kwargs.pop("lu_solver_method")

    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()

        solver = self.block_helper.adjoint_solver
        if solver is None:
            if self.assemble_system:
                rhs_bcs_form = backend.inner(backend.Function(self.function_space),
                                             dFdu_adj_form.arguments()[0]) * backend.dx
                A, _ = backend.assemble_system(dFdu_adj_form, rhs_bcs_form, bcs, **self.assemble_kwargs)
            else:
                A = compat.assemble_adjoint_value(dFdu_adj_form, **self.assemble_kwargs)
                [bc.apply(A) for bc in bcs]
            if self.ident_zeros_tol is not None:
                A.ident_zeros(self.ident_zeros_tol)
            solver = backend.LUSolver(A, self.method)
            self.block_helper.adjoint_solver = solver

        solver.parameters.update(self.lu_solver_parameters)
        [bc.apply(dJdu) for bc in bcs]

        adj_sol = backend.Function(self.function_space)
        solver.solve(adj_sol.vector(), dJdu)

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = compat.function_from_vector(self.function_space, dJdu_copy - compat.assemble_adjoint_value(
                backend.action(dFdu_adj_form, adj_sol)))

        return adj_sol, adj_sol_bdy

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        solver = self.block_helper.forward_solver
        if solver is None:
            if self.assemble_system:
                A, _ = backend.assemble_system(lhs, rhs, bcs, **self.assemble_kwargs)
            else:
                A = compat.assemble_adjoint_value(lhs, **self.assemble_kwargs)
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

        if self.ident_zeros_tol is not None:
            A.ident_zeros(self.ident_zeros_tol)

        solver.parameters.update(self.lu_solver_parameters)
        solver.solve(func.vector(), b)
        return func
