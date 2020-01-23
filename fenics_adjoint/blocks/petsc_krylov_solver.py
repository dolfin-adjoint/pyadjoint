import backend
from . import SolveLinearSystemBlock
from fenics_adjoint.types import as_backend_type
from dolfin_adjoint_common import compat

compat = compat.compat(backend)


class PETScKrylovSolveBlockHelper(object):
    def __init__(self):
        self.forward_solver = None
        self.adjoint_solver = None

    def reset(self):
        self.forward_solver = None
        self.adjoint_solver = None


class PETScKrylovSolveBlock(SolveLinearSystemBlock):
    def __init__(self, A, u, b, *args, **kwargs):
        super(PETScKrylovSolveBlock, self).__init__(A, u, b, **kwargs)
        self.krylov_solver_parameters = kwargs.pop("krylov_solver_parameters")
        self.block_helper = kwargs.pop("block_helper")
        self.pc_operator = kwargs.pop("pc_operator")
        self.nonzero_initial_guess = kwargs.pop("nonzero_initial_guess")
        self.method = kwargs.pop("krylov_method")
        self.preconditioner = kwargs.pop("krylov_preconditioner")
        self.ksp_options_prefix = kwargs.pop("ksp_options_prefix")
        self._ad_nullspace = kwargs.pop("_ad_nullspace")

        if self.nonzero_initial_guess:
            # Here we store a variable that isn't necessarily a dependency.
            # This means that the graph does not know that we depend on this BlockVariable.
            # This could lead to unexpected behaviour in the future.
            # TODO: Consider if this is really a problem.
            self.func.block_variable.save_output()
            self.initial_guess = self.func.block_variable

        if self.pc_operator is not None:
            self.pc_operator = self.pc_operator.form
            for c in self.pc_operator.coefficients():
                self.add_dependency(c)

    def _create_initial_guess(self):
        r = super(PETScKrylovSolveBlock, self)._create_initial_guess()
        if self.nonzero_initial_guess:
            backend.Function.assign(r, self.initial_guess.saved_output)
        return r

    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()

        solver = self.block_helper.adjoint_solver
        if solver is None:
            solver = backend.PETScKrylovSolver(self.method, self.preconditioner)
            solver.ksp().setOptionsPrefix(self.ksp_options_prefix)
            solver.set_from_options()

            if self.assemble_system:
                rhs_bcs_form = backend.inner(backend.Function(self.function_space),
                                             dFdu_adj_form.arguments()[0]) * backend.dx
                A, _ = backend.assemble_system(dFdu_adj_form, rhs_bcs_form, bcs)

                if self._ad_nullspace is not None:
                    as_backend_type(A).set_nullspace(self._ad_nullspace)

                if self.pc_operator is not None:
                    P = self._replace_form(self.pc_operator)
                    P, _ = backend.assemble_system(P, rhs_bcs_form, bcs)
                    solver.set_operators(A, P)
                else:
                    solver.set_operator(A)
            else:
                A = compat.assemble_adjoint_value(dFdu_adj_form)
                [bc.apply(A) for bc in bcs]

                if self._ad_nullspace is not None:
                    as_backend_type(A).set_nullspace(self._ad_nullspace)

                if self.pc_operator is not None:
                    P = self._replace_form(self.pc_operator)
                    P = compat.assemble_adjoint_value(P)
                    [bc.apply(P) for bc in bcs]
                    solver.set_operators(A, P)
                else:
                    solver.set_operator(A)

            self.block_helper.adjoint_solver = solver

        solver.parameters.update(self.krylov_solver_parameters)
        [bc.apply(dJdu) for bc in bcs]

        if self._ad_nullspace is not None:
            if self._ad_nullspace._ad_orthogonalized:
                self._ad_nullspace.orthogonalize(dJdu)

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
            solver = backend.PETScKrylovSolver(self.method, self.preconditioner)
            solver.ksp().setOptionsPrefix(self.ksp_options_prefix)
            solver.set_from_options()

            if self.assemble_system:
                A, _ = backend.assemble_system(lhs, rhs, bcs)
                if self._ad_nullspace is not None:
                    as_backend_type(A).set_nullspace(self._ad_nullspace)

                if self.pc_operator is not None:
                    P = self._replace_form(self.pc_operator)
                    P, _ = backend.assemble_system(P, rhs, bcs)
                    solver.set_operators(A, P)
                else:
                    solver.set_operator(A)
            else:
                A = compat.assemble_adjoint_value(lhs)
                [bc.apply(A) for bc in bcs]
                if self._ad_nullspace is not None:
                    as_backend_type(A).set_nullspace(self._ad_nullspace)

                if self.pc_operator is not None:
                    P = self._replace_form(self.pc_operator)
                    P = compat.assemble_adjoint_value(P)
                    [bc.apply(P) for bc in bcs]
                    solver.set_operators(A, P)
                else:
                    solver.set_operator(A)
            self.block_helper.forward_solver = solver

        if self.assemble_system:
            system_assembler = backend.SystemAssembler(lhs, rhs, bcs)
            b = backend.Function(self.function_space).vector()
            system_assembler.assemble(b)
        else:
            b = compat.assemble_adjoint_value(rhs)
            [bc.apply(b) for bc in bcs]

        if self._ad_nullspace is not None:
            if self._ad_nullspace._ad_orthogonalized:
                self._ad_nullspace.orthogonalize(b)

        solver.parameters.update(self.krylov_solver_parameters)
        solver.solve(func.vector(), b)
        return func
