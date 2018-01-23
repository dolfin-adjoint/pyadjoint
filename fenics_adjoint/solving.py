import backend
import ufl
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from pyadjoint.block import Block
from .types import Function, DirichletBC
from .types import compat
from .types.function_space import extract_subfunction

# Type dependencies

# TODO: Clean up: some inaccurate comments. Reused code. Confusing naming with dFdm when denoting the control as c.


def solve(*args, **kwargs):
    '''This solve routine wraps the real Dolfin solve call. Its purpose is to annotate the model,
    recording what solves occur and what forms are involved, so that the adjoint and tangent linear models may be
    constructed automatically by pyadjoint.

    To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
    Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
    for the purposes of the adjoint computation (such as projecting fields to other function spaces
    for the purposes of visualisation).'''

    annotate = annotate_tape(kwargs)

    if annotate:
        tape = get_working_tape()
        block = SolveBlock(*args, **kwargs)
        tape.add_block(block)

    with stop_annotating():
        output = backend.solve(*args, **kwargs)

    if annotate:
        if hasattr(args[1], "create_block_variable"):
            block_variable = args[1].create_block_variable()
        else:
            block_variable = args[1].function.create_block_variable()
        block.add_output(block_variable)

    return output


class SolveBlock(Block):
    def __init__(self, *args, **kwargs):
        super(SolveBlock, self).__init__()
        if isinstance(args[0], ufl.equation.Equation):
            # Variational problem.
            eq = args[0]
            self.lhs = eq.lhs
            self.rhs = eq.rhs
            self.func = args[1]
            self.kwargs = kwargs

            if len(args) > 2:
                self.bcs = args[2]
            elif "bcs" in kwargs:
                self.bcs = self.kwargs.pop("bcs")
            else:
                self.bcs = []

            # make sure self.bcs is always a list
            if self.bcs is None:
                self.bcs = []

            self.forward_kwargs = kwargs.copy()
            if "J" in self.kwargs:
                self.kwargs["J"] = backend.adjoint(self.kwargs["J"])
            if "Jp" in self.kwargs:
                self.kwargs["Jp"] = backend.adjoint(self.kwargs["Jp"])

            if "M" in self.kwargs:
                raise NotImplemented("Annotation of adaptive solves not implemented.")

            # Some arguments need passing to assemble:
            self.assemble_kwargs = {}
            if "solver_parameters" in kwargs and "mat_type" in kwargs["solver_parameters"]:
                self.assemble_kwargs["mat_type"] = kwargs["solver_parameters"]["mat_type"]

            #self.add_output(self.func.create_block_variable())
        else:
            # Linear algebra problem.
            # TODO: Consider checking if attributes exist.
            A = args[0]
            u = args[1]
            b = args[2]

            self.lhs = A.form
            self.rhs = b.form
            self.bcs = A.bcs if hasattr(A, "bcs") else []
            self.func = u.function

            self.kwargs = kwargs
            self.forward_kwargs = kwargs.copy()
            self.assemble_kwargs = {}

        if not isinstance(self.bcs, list):
            self.bcs = [self.bcs]

        if isinstance(self.lhs, ufl.Form) and isinstance(self.rhs, ufl.Form):
            self.linear = True
            # Add dependence on coefficients on the right hand side.
            for c in self.rhs.coefficients():
                self.add_dependency(c.block_variable)
        else:
            self.linear = False

        for bc in self.bcs:
            self.add_dependency(bc.block_variable)

        for c in self.lhs.coefficients():
            self.add_dependency(c.block_variable)

        self.function_space = self.func.function_space()
        self.adj_sol = None

    def __str__(self):
        return "{} = {}".format(str(self.lhs), str(self.rhs))

    def _create_F_form(self):
        # Process the equation forms, replacing values with checkpoints,
        # and gathering lhs and rhs in one single form.
        if self.linear:
            tmp_u = Function(self.function_space)
            F_form = backend.action(self.lhs, tmp_u) - self.rhs
        else:
            tmp_u = self.func
            F_form = self.lhs

        replace_map = self._replace_map(F_form)
        replace_map[tmp_u] = self.get_outputs()[0].saved_output
        return backend.replace(F_form, replace_map)

    def _replace_map(self, form):
        replace_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in form.coefficients():
                replace_coeffs[coeff] = block_variable.saved_output
        return replace_coeffs

    def _homogenize_bcs(self):
        bcs = []
        for bc in self.bcs:
            if isinstance(bc, backend.DirichletBC):
                bc = compat.create_bc(bc, homogenize=True)
            bcs.append(bc)
        return bcs

    def _assemble_and_solve_adj_eq(self, dFdu_form, dJdu):
        dJdu_copy = dJdu.copy()
        dFdu = compat.assemble_adjoint_value(dFdu_form, **self.assemble_kwargs)

        # Homogenize and apply boundary conditions on adj_dFdu and dJdu.
        for bc in self._homogenize_bcs():
            bc.apply(dFdu, dJdu)

        adj_sol = Function(self.function_space)
        backend.solve(dFdu, adj_sol.vector(), dJdu)

        adj_sol_bdy = compat.function_from_vector(self.function_space, dJdu_copy - compat.assemble_adjoint_value(
            backend.action(dFdu_form, adj_sol)))

        return adj_sol, adj_sol_bdy

    @no_annotations
    def evaluate_adj(self):
        # Get dJdu from previous calculations.
        dJdu = self.get_outputs()[0].adj_value

        if dJdu is None:
            return

        F_form = self._create_F_form()

        dFdu = backend.derivative(F_form, self.get_outputs()[0].saved_output, backend.TrialFunction(self.function_space))
        dFdu_form = backend.adjoint(dFdu)
        # Copy to make sure bc.apply doesn't modify adj_value.
        dJdu = dJdu.copy()

        adj_sol, adj_sol_bdy = self._assemble_and_solve_adj_eq(dFdu_form, dJdu)
        self.adj_sol = adj_sol

        for block_variable in self.get_dependencies():
            c = block_variable.output
            if c != self.func or self.linear:
                # c_rep = replaced_coeffs.get(c, c)
                c_rep = block_variable.saved_output

                if isinstance(c, backend.Function):
                    dFdm = -backend.derivative(F_form, c_rep, backend.TrialFunction(c.function_space()))
                    dFdm = backend.adjoint(dFdm)
                    dFdm = dFdm*adj_sol
                    dFdm = compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)

                    block_variable.add_adj_output(dFdm)
                elif isinstance(c, backend.Constant):
                    mesh = compat.extract_mesh_from_form(F_form)
                    dFdm = -backend.derivative(F_form, c_rep, backend.TrialFunction(c._ad_function_space(mesh)))
                    dFdm = backend.adjoint(dFdm)
                    dFdm = dFdm*adj_sol
                    dFdm = compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)

                    block_variable.add_adj_output(dFdm)
                elif isinstance(c, backend.DirichletBC):
                    tmp_bc = compat.create_bc(c, value=extract_subfunction(adj_sol_bdy, c.function_space()))
                    block_variable.add_adj_output([tmp_bc])
                elif isinstance(c, backend.Expression):
                    mesh = F_form.ufl_domain().ufl_cargo()
                    c_fs = c._ad_function_space(mesh)
                    dFdm = -backend.derivative(F_form, c_rep, backend.TrialFunction(c_fs))
                    dFdm = backend.adjoint(dFdm)
                    dFdm = dFdm * adj_sol
                    dFdm = compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)
                    block_variable.add_adj_output([[dFdm, c_fs]])

    @no_annotations
    def evaluate_tlm(self):
        F_form = self._create_F_form()

        # Obtain dFdu.
        dFdu = backend.derivative(F_form, self.get_outputs()[0].saved_output, backend.TrialFunction(self.function_space))

        dFdu = backend.assemble(dFdu, **self.assemble_kwargs)

        # Homogenize and apply boundary conditions on dFdu.
        bcs = []
        for bc in self.bcs:
            if isinstance(bc, backend.DirichletBC):
                bc = compat.create_bc(bc, homogenize=True)
            bcs.append(bc)
            bc.apply(dFdu)

        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            if tlm_value is None:
                continue

            c = block_variable.output
            c_rep = block_variable.saved_output

            if c == self.func and not self.linear:
                continue

            if isinstance(c, backend.Function):
                # TODO: If tlm_value is a Sum, will this crash in some instances? Should we project?
                dFdm = -backend.derivative(F_form, c_rep, tlm_value)
                dFdm = compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)

                # Zero out boundary values from boundary conditions as they do not depend (directly) on c.
                for bc in bcs:
                    bc.apply(dFdm)

            elif isinstance(c, backend.Constant):
                dFdm = -backend.derivative(F_form, c_rep, tlm_value)
                dFdm = compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)

                # Zero out boundary values from boundary conditions as they do not depend (directly) on c.
                for bc in bcs:
                    bc.apply(dFdm)

            elif isinstance(c, backend.DirichletBC):
                #tmp_bc = backend.DirichletBC(V, tlm_value, c_rep.user_sub_domain())
                dFdm = backend.Function(self.function_space).vector()
                tlm_value.apply(dFdu, dFdm)

            elif isinstance(c, backend.Expression):
                dFdm = -backend.derivative(F_form, c_rep, tlm_value)
                dFdm = compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)

                # Zero out boundary values from boundary conditions as they do not depend (directly) on c.
                for bc in bcs:
                    bc.apply(dFdm)

            dudm = Function(self.function_space)
            backend.solve(dFdu, dudm.vector(), dFdm)

            self.get_outputs()[0].add_tlm_output(dudm)

    def _assemble_and_solve_soa_eq(self, dFdu_form, adj_sol, hessian_input, d2Fdu2):
        dFdu = compat.assemble_adjoint_value(dFdu_form, **self.assemble_kwargs)

        # Start piecing together the rhs of the soa equation
        b = hessian_input.copy()
        b_form = d2Fdu2

        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value

            if (c == self.func and not self.linear) or tlm_input is None:
                continue

            if not isinstance(c, backend.DirichletBC):
                d2Fdudm = ufl.algorithms.expand_derivatives(backend.derivative(dFdu_form, c_rep, tlm_input))
                b_form += d2Fdudm

        if len(b_form.integrals()) > 0:
            b_form = backend.adjoint(b_form)
            b -= compat.assemble_adjoint_value(backend.action(b_form, adj_sol))
        b_copy = b.copy()

        # Homogenize and apply boundary conditions.
        for bc in self._homogenize_bcs():
            bc.apply(dFdu, b)

        adj_sol2 = Function(self.function_space)
        # Solve the soa equation
        backend.solve(dFdu, adj_sol2.vector(), b)
        adj_sol2_bdy = compat.function_from_vector(self.function_space, b_copy - compat.assemble_adjoint_value(
            backend.action(dFdu_form, adj_sol2)))

        return adj_sol2, adj_sol2_bdy

    @no_annotations
    def evaluate_hessian(self):
        # First fetch all relevant values
        fwd_block_variable = self.get_outputs()[0]
        hessian_input = fwd_block_variable.hessian_value
        tlm_output = fwd_block_variable.tlm_value

        if hessian_input is None:
            return

        if tlm_output is None:
            return

        F_form = self._create_F_form()

        # Using the equation Form we derive dF/du, d^2F/du^2 * du/dm * direction.
        dFdu_form = backend.derivative(F_form, fwd_block_variable.saved_output)
        d2Fdu2 = ufl.algorithms.expand_derivatives(
            backend.derivative(dFdu_form, fwd_block_variable.saved_output, tlm_output))

        dFdu_form = backend.adjoint(dFdu_form)
        adj_sol = self.adj_sol
        if adj_sol is None:
            raise RuntimeError("Hessian computation run before adjoint.")
        adj_sol2, adj_sol2_bdy = self._assemble_and_solve_soa_eq(dFdu_form, adj_sol, hessian_input, d2Fdu2)

        # Iterate through every dependency to evaluate and propagate the hessian information.
        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output

            if c == self.func and not self.linear:
                continue

            # If m = DirichletBC then d^2F(u,m)/dm^2 = 0 and d^2F(u,m)/dudm = 0,
            # so we only have the term dF(u,m)/dm * adj_sol2
            if isinstance(c, backend.DirichletBC):
                tmp_bc = compat.create_bc(c, value=adj_sol2_bdy)
                bo.add_hessian_output([tmp_bc])
                continue

            dc = None
            if isinstance(c_rep, backend.Constant):
                mesh = compat.extract_mesh_from_form(F_form)
                W = c._ad_function_space(mesh)
            elif isinstance(c, backend.Expression):
                mesh = F_form.ufl_domain().ufl_cargo()
                W = c._ad_function_space(mesh)
            else:
                W = c.function_space()

            dc = backend.TrialFunction(W)
            dFdm = backend.derivative(F_form, c_rep, dc)
            # TODO: Actually implement split annotations properly.
            try:
                d2Fdudm = ufl.algorithms.expand_derivatives(backend.derivative(dFdm, fwd_block_variable.saved_output, tlm_output))
            except ufl.log.UFLException:
                continue

            # We need to add terms from every other dependency
            # i.e. the terms d^2F/dm_1dm_2
            for bo2 in self.get_dependencies():
                c2 = bo2.output
                c2_rep = bo2.saved_output

                if isinstance(c2, backend.DirichletBC):
                    continue

                tlm_input = bo2.tlm_value
                if tlm_input is None:
                    continue

                if c2 == self.func and not self.linear:
                    continue

                # TODO: If tlm_input is a Sum, this crashes in some instances?
                d2Fdm2 = ufl.algorithms.expand_derivatives(backend.derivative(dFdm, c2_rep, tlm_input))
                if d2Fdm2.empty():
                    continue

                if len(d2Fdm2.arguments()) >= 2:
                    d2Fdm2 = backend.adjoint(d2Fdm2)

                output = backend.action(d2Fdm2, adj_sol)
                output = compat.assemble_adjoint_value(-output)

                if isinstance(c, backend.Expression):
                    bo.add_hessian_output([(output, W)])
                else:
                    bo.add_hessian_output(output)

            if len(dFdm.arguments()) >= 2:
                dFdm = backend.adjoint(dFdm)
            output = backend.action(dFdm, adj_sol2)
            if not d2Fdudm.empty():
                if len(d2Fdudm.arguments()) >= 2:
                    d2Fdudm = backend.adjoint(d2Fdudm)
                output += backend.action(d2Fdudm, adj_sol)

            output = compat.assemble_adjoint_value(-output)

            if isinstance(c, backend.Expression):
                bo.add_hessian_output([(output, W)])
            else:
                bo.add_hessian_output(output)

    @no_annotations
    def recompute(self):
        if self.get_outputs()[0].is_control:
            return

        func = self.get_outputs()[0].saved_output
        replace_lhs_coeffs = {}
        replace_rhs_coeffs = {}
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, backend.DirichletBC):
                bcs.append(c_rep)
            elif c != c_rep:
                if c in self.lhs.coefficients():
                    replace_lhs_coeffs[c] = c_rep
                    if c == self.func:
                        backend.Function.assign(func, c_rep)
                        replace_lhs_coeffs[c] = func

                if self.linear and c in self.rhs.coefficients():
                    replace_rhs_coeffs[c] = c_rep

        lhs = backend.replace(self.lhs, replace_lhs_coeffs)

        rhs = 0
        if self.linear:
            rhs = backend.replace(self.rhs, replace_rhs_coeffs)

        # Here we overwrite the checkpoint (if nonlin solve). Is this a good idea?
        # In theory should not matter, but may sometimes lead to
        # unexpected results if not careful.
        backend.solve(lhs == rhs, func, bcs, **self.forward_kwargs)
        # Save output for use in later re-computations.
        # TODO: Consider redesigning the saving system so a new deepcopy isn't created on each forward replay.
        self.get_outputs()[0].checkpoint = func._ad_create_checkpoint()

