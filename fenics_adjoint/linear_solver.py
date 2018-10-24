import backend

from .solving import SolveBlock
from .types import Function
from pyadjoint.tape import annotate_tape, get_working_tape
from .types import compat


class LinearSolver(backend.LinearSolver):
    """This object is overloaded so that solves using this class are automatically annotated,
    so that libadjoint can automatically derive the adjoint and tangent linear models."""
    def __init__(self, *args):
        backend.LinearSolver.__init__(self, *args)
        self.solver_parameters = args
        self.nsp = None

        self.operators = (None, None)
        if len(args) > 0 and isinstance(args[0], backend.GenericMatrix):
            self.operators = (args[0], None)

    def set_operators(self, A, P):
        backend.LinearSolver.set_operators(self, A, P)
        self.operators = (A, P)

    def set_nullspace(self, nsp):
        self.nsp = nsp

    def set_operator(self, A):
        backend.LinearSolver.set_operator(self, A)
        self.operators = (A, self.operators[1])

    def solve(self, *args, **kwargs):
        """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
        for the purposes of the adjoint computation (such as projecting fields to other function spaces
        for the purposes of visualisation)."""

        annotate = annotate_tape(kwargs)
        nsp = self.nsp

        if annotate:
            if len(args) == 3:
                A = args[0]
                x = args[1]
                b = args[2]
            elif len(args) == 2:
                A = self.operators[0]
                x = args[0]
                b = args[1]

            if self.operators[1] is not None:
                P = self.operators[1].form
            else:
                P = None

            u = x.function
            solver_parameters = self.solver_parameters
            parameters = self.parameters.to_dict()
            has_preconditioner = P is not None
            nonzero_initial_guess = parameters.get("nonzero_initial_guess", False)

            tape = get_working_tape()
            block = LinearSolveBlock(A, x, b,
                                     ad_block_parameters={"solver_parameters": solver_parameters,
                                                          "parameters": parameters,
                                                          "has_preconditioner": has_preconditioner,
                                                          "P": P,
                                                          "nullspace": nsp,
                                                          "nonzero_initial_guess": nonzero_initial_guess})
            tape.add_block(block)

        out = backend.LinearSolver.solve(self, *args, **kwargs)

        if annotate:
            block.add_output(u.create_block_variable())

        return out


class LinearSolveBlock(SolveBlock):
    def __init__(self, *args, **kwargs):
        super(LinearSolveBlock, self).__init__(*args, **kwargs)
        block_parameters = kwargs.pop("ad_block_parameters")
        self.solver_parameters = block_parameters.pop("solver_parameters")
        self.has_preconditioner = block_parameters.pop("has_preconditioner")
        self.P = block_parameters.pop("P")
        self.parameters = block_parameters.pop("parameters")
        # TODO: nullspace is not used. Make use of it or remove it.
        self.nullspace = block_parameters.pop("nullspace")
        self.nonzero_initial_guess = block_parameters.pop("nonzero_initial_guess")

        if self.nonzero_initial_guess:
            # Here we store a variable that isn't necessarily a dependency.
            # This means that the graph does not know that we depend on this BlockVariable.
            # This could lead to unexpected behaviour in the future.
            # TODO: Consider if this is really a problem.
            self.func.block_variable.save_output()
            self.initial_guess = self.func.block_variable

        if self.has_preconditioner:
            for c in self.P.coefficients():
                self.add_dependency(c.block_variable)

    def _create_initial_guess(self):
        r = super(LinearSolveBlock, self)._create_initial_guess()
        if self.nonzero_initial_guess:
            backend.Function.assign(r, self.initial_guess.saved_output)
        return r

    def _assemble_and_solve_adj_eq(self, dFdu_form, dJdu):
        dJdu_copy = dJdu.copy()
        solver = backend.LinearSolver(*self.solver_parameters)
        solver.parameters.update(self.parameters)
        bcs = self._homogenize_bcs()

        if self.assemble_system:
            # Since dJdu is a vector we can't use it directly in assemble_system.
            [bc.apply(dJdu) for bc in bcs]
            v = backend.TestFunction(self.function_space)
            rhs_form = backend.inner(backend.Function(self.function_space), v)*backend.dx
            dFdu, rhs_bcs = backend.assemble_system(dFdu_form, rhs_form, bcs)
            dJdu += rhs_bcs

            if self.has_preconditioner:
                P = self._replace_form(self.P)
                P, _ = backend.assemble_system(P, rhs_form, bcs)
                solver.set_operators(dFdu, P)
            else:
                solver.set_operator(dFdu)
        else:
            dFdu = compat.assemble_adjoint_value(dFdu_form)
            [bc.apply(dFdu, dJdu) for bc in bcs]

            if self.has_preconditioner:
                P = self._replace_form(self.P)
                P = compat.assemble_adjoint_value(P)
                [bc.apply(P) for bc in bcs]
                solver.set_operators(dFdu, P)
            else:
                solver.set_operator(dFdu)

        adj_sol = Function(self.function_space)
        solver.solve(adj_sol.vector(), dJdu)

        adj_sol_bdy = compat.function_from_vector(self.function_space, dJdu_copy - compat.assemble_adjoint_value(
            backend.action(dFdu_form, adj_sol)))

        return adj_sol, adj_sol_bdy

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        solver = backend.LinearSolver(*self.solver_parameters)
        solver.parameters.update(self.parameters)

        if self.assemble_system:
            A, b = backend.assemble_system(lhs, rhs, bcs)

            if self.has_preconditioner:
                P = self._replace_form(self.P)
                P, _ = backend.assemble_system(P, rhs, bcs)
                solver.set_operators(A, P)
            else:
                solver.set_operator(A)
        else:
            A = compat.assemble_adjoint_value(lhs)
            b = compat.assemble_adjoint_value(rhs)
            [bc.apply(A, b) for bc in bcs]

            if self.has_preconditioner:
                P = self._replace_form(self.P)
                P = compat.assemble_adjoint_value(P)
                [bc.apply(P) for bc in bcs]
                solver.set_operators(A, P)
            else:
                solver.set_operator(A)

        solver.solve(func.vector(), b)
        return func


