import backend
import ufl
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from pyadjoint.block import Block
from .types import Function, DirichletBC
from .types import compat
from .types.function_space import extract_subfunction
import numpy

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
        self.adj_sol = None
        self.varform = isinstance(args[0], ufl.equation.Equation)
        self._init_solver_parameters(*args, **kwargs)
        self._init_dependencies(*args, **kwargs)
        self.function_space = self.func.function_space()

    def __str__(self):
        return "{} = {}".format(str(self.lhs), str(self.rhs))

    def _init_solver_parameters(self, *args, **kwargs):
        if self.varform:
            self.kwargs = kwargs
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
        else:
            self.kwargs = kwargs
            self.forward_kwargs = kwargs.copy()
            self.assemble_kwargs = {}

    def _init_dependencies(self, *args, **kwargs):
        if self.varform:
            eq = args[0]
            self.lhs = eq.lhs
            self.rhs = eq.rhs
            self.func = args[1]

            if len(args) > 2:
                self.bcs = args[2]
            elif "bcs" in kwargs:
                self.bcs = self.kwargs.pop("bcs")
            else:
                self.bcs = []

            if self.bcs is None:
                self.bcs = []

            self.assemble_system = False
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
            self.assemble_system = A.assemble_system if hasattr(A, "assemble_system") else False

        if not isinstance(self.bcs, list):
            self.bcs = [self.bcs]

        if isinstance(self.lhs, ufl.Form) and isinstance(self.rhs, ufl.Form):
            self.linear = True
            # Add dependence on coefficients on the right hand side.
            for c in self.rhs.coefficients():
                self.add_dependency(c.block_variable, no_duplicates=True)
        else:
            self.linear = False

        for bc in self.bcs:
            self.add_dependency(bc.block_variable, no_duplicates=True)

        for c in self.lhs.coefficients():
            self.add_dependency(c.block_variable, no_duplicates=True)

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
        return ufl.replace(F_form, replace_map)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        fwd_block_variable = self.get_outputs()[0]
        u = fwd_block_variable.output

        dJdu = adj_inputs[0]

        F_form = self._create_F_form()

        dFdu = backend.derivative(F_form, fwd_block_variable.saved_output, backend.TrialFunction(u.function_space()))
        dFdu_form = backend.adjoint(dFdu)
        dJdu = dJdu.copy()
        adj_sol, adj_sol_bdy = self._assemble_and_solve_adj_eq(dFdu_form, dJdu)
        self.adj_sol = adj_sol

        r = {}
        r["form"] = F_form
        r["adj_sol"] = adj_sol
        r["adj_sol_bdy"] = adj_sol_bdy
        return r

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

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if not self.linear and self.func == block_variable.output:
            # We are not able to calculate derivatives wrt initial guess.
            return None
        F_form = prepared["form"]
        adj_sol = prepared["adj_sol"]
        adj_sol_bdy = prepared["adj_sol_bdy"]
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, backend.Function):
            trial_function = backend.TrialFunction(c.function_space())
        elif isinstance(c, backend.Constant):
            mesh = compat.extract_mesh_from_form(F_form)
            trial_function = backend.TrialFunction(c._ad_function_space(mesh))
        elif isinstance(c, compat.ExpressionType):
            mesh = F_form.ufl_domain().ufl_cargo()
            c_fs = c._ad_function_space(mesh)
            trial_function = backend.TrialFunction(c_fs)
        elif isinstance(c, backend.DirichletBC):
            tmp_bc = compat.create_bc(c, value=extract_subfunction(adj_sol_bdy, c.function_space()))
            return [tmp_bc]

        dFdm = -backend.derivative(F_form, c_rep, trial_function)
        dFdm = backend.adjoint(dFdm)
        dFdm = dFdm * adj_sol
        dFdm = compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)
        if isinstance(c, compat.ExpressionType):
            return [[dFdm, c_fs]]
        else:
            return dFdm

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        fwd_block_variable = self.get_outputs()[0]
        u = fwd_block_variable.output

        F_form = self._create_F_form()

        # Obtain dFdu.
        dFdu = backend.derivative(F_form, fwd_block_variable.saved_output, backend.TrialFunction(u.function_space()))

        r = {}
        r["form"] = F_form
        r["dFdu"] = dFdu
        return r

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        F_form = prepared["form"]
        dFdu = prepared["dFdu"]
        V = self.get_outputs()[idx].output.function_space()

        bcs = []
        dFdm = 0.
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, backend.DirichletBC):
                if tlm_value is None:
                    bcs.append(compat.create_bc(c, homogenize=True))
                else:
                    bcs.append(tlm_value)
                continue
            if tlm_value is None:
                continue

            if c == self.func and not self.linear:
                continue

            dFdm += -backend.derivative(F_form, c_rep, tlm_value)

        if isinstance(dFdm, float):
            v = dFdu.arguments()[0]
            dFdm = backend.inner(backend.Constant(numpy.zeros(v.ufl_shape)), v)*backend.dx

        dudm = backend.Function(V)
        return self._assemble_and_solve_tlm_eq(dFdu, dFdm, dudm, bcs)

    def _assemble_and_solve_tlm_eq(self, dFdu, dFdm, dudm, bcs):
        return self._forward_solve(dFdu, dFdm, dudm, bcs)

    def _assemble_soa_eq_rhs(self, dFdu_form, adj_sol, hessian_input, d2Fdu2):
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
                d2Fdudm = backend.derivative(dFdu_form, c_rep, tlm_input)
                b_form += d2Fdudm

        b_form = ufl.algorithms.expand_derivatives(b_form)
        if len(b_form.integrals()) > 0:
            b_form = backend.adjoint(b_form)
            b -= compat.assemble_adjoint_value(backend.action(b_form, adj_sol))

        return b

    def _assemble_and_solve_soa_eq(self, dFdu_form, adj_sol, hessian_input, d2Fdu2):
        b = self._assemble_soa_eq_rhs(dFdu_form, adj_sol, hessian_input, d2Fdu2)
        adj_sol2, adj_sol2_bdy = self._assemble_and_solve_adj_eq(dFdu_form, b)
        return adj_sol2, adj_sol2_bdy

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        # First fetch all relevant values
        fwd_block_variable = self.get_outputs()[0]
        adj_input = adj_inputs[0]
        hessian_input = hessian_inputs[0]
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
            raise RuntimeError("Hessian computation was run before adjoint.")
        adj_sol2, adj_sol2_bdy = self._assemble_and_solve_soa_eq(dFdu_form, adj_sol, hessian_input, d2Fdu2)

        r = {}
        r["adj_sol2"] = adj_sol2
        r["adj_sol2_bdy"] = adj_sol2_bdy
        r["form"] = F_form
        r["adj_sol"] = adj_sol
        return r

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        c = block_variable.output
        if c == self.func and not self.linear:
            return None

        adj_sol2 = prepared["adj_sol2"]
        adj_sol2_bdy = prepared["adj_sol2_bdy"]
        F_form = prepared["form"]
        adj_sol = prepared["adj_sol"]
        fwd_block_variable = self.get_outputs()[0]
        tlm_output = fwd_block_variable.tlm_value

        c_rep = block_variable.saved_output

        # If m = DirichletBC then d^2F(u,m)/dm^2 = 0 and d^2F(u,m)/dudm = 0,
        # so we only have the term dF(u,m)/dm * adj_sol2
        if isinstance(c, backend.DirichletBC):
            tmp_bc = compat.create_bc(c, value=extract_subfunction(adj_sol2_bdy, c.function_space()))
            return [tmp_bc]

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
        # TODO: Old comment claims this might break on split. Confirm if true or not.
        d2Fdudm = ufl.algorithms.expand_derivatives(backend.derivative(dFdm, fwd_block_variable.saved_output, tlm_output))

        hessian_output_form = 0

        # We need to add terms from every other dependency
        # i.e. the terms d^2F/dm_1dm_2
        for _, bv in relevant_dependencies:
            c2 = bv.output
            c2_rep = bv.saved_output

            if isinstance(c2, backend.DirichletBC):
                continue

            tlm_input = bv.tlm_value
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
            hessian_output_form += -output

        if len(dFdm.arguments()) >= 2:
            dFdm = backend.adjoint(dFdm)
        output = backend.action(dFdm, adj_sol2)
        if not d2Fdudm.empty():
            if len(d2Fdudm.arguments()) >= 2:
                d2Fdudm = backend.adjoint(d2Fdudm)
            output += backend.action(d2Fdudm, adj_sol)
        hessian_output_form += -output

        hessian_output = compat.assemble_adjoint_value(hessian_output_form)
        if isinstance(c, backend.Expression):
            return [(hessian_output, W)]
        else:
            return hessian_output

    def _create_initial_guess(self):
        return backend.Function(self.function_space)

    def _replace_recompute_form(self):
        func = self._create_initial_guess()

        bcs = self._recover_bcs()
        lhs = self._replace_form(self.lhs, func=func)
        rhs = 0
        if self.linear:
            rhs = self._replace_form(self.rhs)

        return lhs, rhs, func, bcs

    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, backend.DirichletBC):
                bcs.append(c_rep)
        return bcs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        lhs, rhs, func, bcs = self._replace_recompute_form()
        return lhs, rhs, func, bcs

    def _replace_form(self, form, func=None):
        replace_map = {}
        for block_variable in self.get_dependencies():
            c = block_variable.output
            if c in form.coefficients():
                c_rep = block_variable.saved_output
                if c != c_rep:
                    replace_map[c] = c_rep
                    if func is not None and c == self.func:
                        backend.Function.assign(func, c_rep)
                        replace_map[c] = func
        return ufl.replace(form, replace_map)

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        backend.solve(lhs == rhs, func, bcs, **kwargs)
        return func

    def recompute_component(self, inputs, block_variable, idx, prepared):
        lhs = prepared[0]
        rhs = prepared[1]
        func = prepared[2]
        bcs = prepared[3]

        return self._forward_solve(lhs, rhs, func, bcs, **self.forward_kwargs)
