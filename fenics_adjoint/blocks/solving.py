from . import GenericSolveBlock


class SolveLinearSystemBlock(GenericSolveBlock):
    def __init__(self, A, u, b, *args, **kwargs):
        lhs = A.form
        func = u.function
        rhs = b.form
        bcs = A.bcs if hasattr(A, "bcs") else []
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)

        # Set up parameters initialization
        self.assemble_kwargs["keep_diagonal"] = A.keep_diagonal if hasattr(A, "keep_diagonal") else False
        self.ident_zeros_tol = A.ident_zeros_tol if hasattr(A, "ident_zeros_tol") else None
        self.assemble_system = A.assemble_system if hasattr(A, "assemble_system") else False

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        if len(self.forward_args) <= 0:
            self.forward_args = args

        if len(self.adj_args) <= 0:
            self.adj_args = self.forward_args

    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()
        if self.assemble_system:
            rhs_bcs_form = self.backend.inner(self.backend.Function(self.function_space),
                                              dFdu_adj_form.arguments()[0]) * self.backend.dx
            A, _ = self.backend.assemble_system(dFdu_adj_form, rhs_bcs_form, bcs, **self.assemble_kwargs)
        else:
            kwargs = self.assemble_kwargs.copy()
            kwargs["bcs"] = bcs
            A = self.compat.assemble_adjoint_value(dFdu_adj_form, **kwargs)
        if self.ident_zeros_tol is not None:
            A.ident_zeros(self.ident_zeros_tol)
        [bc.apply(dJdu) for bc in bcs]

        adj_sol = self.compat.create_function(self.function_space)
        self.compat.linalg_solve(A, adj_sol.vector(), dJdu, *self.adj_args, **self.adj_kwargs)

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = self.compat.function_from_vector(self.function_space,
                                                           dJdu_copy - self.compat.assemble_adjoint_value(
                                                               self.backend.action(dFdu_adj_form, adj_sol)))

        return adj_sol, adj_sol_bdy

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        if self.assemble_system:
            A, b = self.backend.assemble_system(lhs, rhs, bcs)
        else:
            assemble_kwargs = self.assemble_kwargs.copy()
            assemble_kwargs["bcs"] = bcs
            A = self.compat.assemble_adjoint_value(lhs, **assemble_kwargs)
            b = self.backend.assemble(rhs)
            [bc.apply(b) for bc in bcs]

        if self.ident_zeros_tol is not None:
            A.ident_zeros(self.ident_zeros_tol)

        self.backend.solve(A, func.vector(), b, *self.forward_args, **self.forward_kwargs)
        return func


class SolveVarFormBlock(GenericSolveBlock):
    pop_kwargs_keys = GenericSolveBlock.pop_kwargs_keys

    def __init__(self, equation, func, bcs=[], *args, **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        if len(self.forward_args) <= 0:
            self.forward_args = args

        if len(self.forward_kwargs) <= 0:
            self.forward_kwargs = kwargs

        if "solver_parameters" in self.forward_kwargs and "mat_type" in self.forward_kwargs["solver_parameters"]:
            self.assemble_kwargs["mat_type"] = self.forward_kwargs["solver_parameters"]["mat_type"]

        if len(self.adj_kwargs) <= 0:
            solver_parameters = kwargs.get("solver_parameters", {})
            if len(self.adj_args) <= 0:
                if "linear_solver" in solver_parameters:
                    adj_args = [solver_parameters["linear_solver"]]
                    if "preconditioner" in solver_parameters:
                        adj_args.append(solver_parameters["preconditioner"])
                    self.adj_args = tuple(adj_args)
                elif "newton_solver" in solver_parameters and "linear_solver" in solver_parameters["newton_solver"]:
                    adj_args = [solver_parameters["newton_solver"]["linear_solver"]]
                    if "preconditioner" in solver_parameters["newton_solver"]:
                        adj_args.append(solver_parameters["newton_solver"]["preconditioner"])
                    self.adj_args = tuple(adj_args)
            self.adj_kwargs = solver_parameters

    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy=True):
        dJdu_copy = dJdu.copy()
        bcs = self._homogenize_bcs()
        kwargs = self.assemble_kwargs.copy()
        kwargs["bcs"] = bcs
        dFdu = self.compat.assemble_adjoint_value(dFdu_adj_form, **kwargs)

        # Apply boundary conditions on adj_dFdu and dJdu.
        for bc in bcs:
            bc.apply(dJdu)

        adj_sol = self.compat.create_function(self.function_space)
        lu_solver_methods = self.backend.lu_solver_methods()
        solver_method = self.adj_args[0] if len(self.adj_args) >= 1 else "default"
        solver_method = "default" if solver_method == "lu" else solver_method

        if solver_method in lu_solver_methods:
            solver = self.backend.LUSolver(solver_method)
            solver_parameters = self.adj_kwargs.get("lu_solver", {})
        else:
            solver = self.backend.KrylovSolver(*self.adj_args)
            solver_parameters = self.adj_kwargs.get("krylov_solver", {})
        solver.parameters.update(solver_parameters)
        solver.solve(dFdu, adj_sol.vector(), dJdu)

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = self.compat.function_from_vector(self.function_space,
                                                           dJdu_copy - self.compat.assemble_adjoint_value(
                                                               self.backend.action(dFdu_adj_form, adj_sol)))

        return adj_sol, adj_sol_bdy
