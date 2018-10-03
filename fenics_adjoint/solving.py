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
                self.add_dependency(c.block_variable, no_duplicates=True)
        else:
            self.linear = False

        for bc in self.bcs:
            self.add_dependency(bc.block_variable, no_duplicates=True)

        for c in self.lhs.coefficients():
            self.add_dependency(c.block_variable, no_duplicates=True)

    def __str__(self):
        return "{} = {}".format(str(self.lhs), str(self.rhs))

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        fwd_block_variable = self.get_outputs()[0]
        u = fwd_block_variable.output
        V = u.function_space()
        adj_var = Function(V)

        dJdu = adj_inputs[0]

        if self.linear:
            tmp_u = Function(V)
            F_form = backend.action(self.lhs, tmp_u) - self.rhs
        else:
            tmp_u = self.func
            F_form = self.lhs

        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in F_form.coefficients():
                replaced_coeffs[coeff] = block_variable.saved_output
        replaced_coeffs[tmp_u] = fwd_block_variable.saved_output

        F_form = ufl.replace(F_form, replaced_coeffs)

        dFdu = backend.derivative(F_form, fwd_block_variable.saved_output, backend.TrialFunction(u.function_space()))
        dFdu_form = backend.adjoint(dFdu)
        dFdu = compat.assemble_adjoint_value(dFdu_form, **self.assemble_kwargs)

        dJdu = dJdu.copy()
        dJdu_copy = dJdu.copy()

        # Homogenize and apply boundary conditions on adj_dFdu and dJdu.
        bcs = []
        for bc in self.bcs:
            if isinstance(bc, backend.DirichletBC):
                bc = compat.create_bc(bc, homogenize=True)
            bcs.append(bc)
            bc.apply(dFdu, dJdu)

        compat.linalg_solve(dFdu, adj_var.vector(), dJdu, **self.kwargs)
        adj_var_bdy = compat.function_from_vector(V, dJdu_copy - compat.assemble_adjoint_value(backend.action(dFdu_form, adj_var)))

        r = {}
        r["form"] = F_form
        r["adj_sol"] = adj_var
        r["adj_sol_bdy"] = adj_var_bdy
        r["coeffs"] = replaced_coeffs
        return r

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if not self.linear and self.func == block_variable.output:
            # We are not able to calculate derivatives wrt initial guess.
            return None
        F_form = prepared["form"]
        adj_sol = prepared["adj_sol"]
        adj_sol_bdy = prepared["adj_sol_bdy"]
        coeffs = prepared["coeffs"]
        c = block_variable.output
        c_rep = coeffs.get(c, c)

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

        if self.linear:
            tmp_u = Function(self.func.function_space()) # Replace later? Maybe save function space on initialization.
            F_form = backend.action(self.lhs, tmp_u) - self.rhs
        else:
            tmp_u = self.func
            F_form = self.lhs

        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in F_form.coefficients():
                replaced_coeffs[coeff] = block_variable.saved_output

        replaced_coeffs[tmp_u] = fwd_block_variable.saved_output

        F_form = ufl.replace(F_form, replaced_coeffs)

        # Obtain dFdu.
        dFdu = backend.derivative(F_form, fwd_block_variable.saved_output, backend.TrialFunction(u.function_space()))
        dFdu = backend.assemble(dFdu, **self.assemble_kwargs)

        r = {}
        r["form"] = F_form
        r["dFdu"] = dFdu
        r["coeffs"] = replaced_coeffs
        return r

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        F_form = prepared["form"]
        dFdu = prepared["dFdu"]
        replaced_coeffs = prepared["coeffs"]
        V = self.get_outputs()[idx].output.function_space()

        bcs = []
        dFdm = 0.
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c = block_variable.output

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

            c_rep = replaced_coeffs.get(c, c)
            dFdm += -backend.derivative(F_form, c_rep, tlm_value)

        if isinstance(dFdm, float):
            dFdm = backend.Function(V).vector()
        else:
            dFdm = compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)

        dudm = backend.Function(V)
        [bc.apply(dFdu, dFdm) for bc in bcs]
        compat.linalg_solve(dFdu, dudm.vector(), dFdm, **self.kwargs)
        return dudm

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        # First fetch all relevant values
        fwd_block_variable = self.get_outputs()[0]
        adj_input = adj_inputs[0]
        hessian_input = hessian_inputs[0]
        tlm_output = fwd_block_variable.tlm_value
        u = fwd_block_variable.output
        V = u.function_space()

        # Process the equation forms, replacing values with checkpoints,
        # and gathering lhs and rhs in one single form.
        if self.linear:
            tmp_u = Function(self.func.function_space())  # Replace later? Maybe save function space on initialization.
            F_form = backend.action(self.lhs, tmp_u) - self.rhs
        else:
            tmp_u = self.func
            F_form = self.lhs

        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in F_form.coefficients():
                replaced_coeffs[coeff] = block_variable.saved_output

        replaced_coeffs[tmp_u] = fwd_block_variable.saved_output
        F_form = ufl.replace(F_form, replaced_coeffs)

        # Define the equation Form. This class is an initial step in refactoring
        # the SolveBlock methods.
        F = Form(F_form, transpose=True)
        F.set_boundary_conditions(self.bcs, fwd_block_variable.saved_output)

        bcs = F.bcs

        # Using the equation Form we derive dF/du, d^2F/du^2 * du/dm * direction.
        dFdu_form = backend.derivative(F_form, fwd_block_variable.saved_output)
        d2Fdu2 = ufl.algorithms.expand_derivatives(
            backend.derivative(dFdu_form, fwd_block_variable.saved_output, tlm_output))

        dFdu = backend.adjoint(dFdu_form)
        dFdu = backend.assemble(dFdu, **self.assemble_kwargs)

        for bc in bcs:
            bc.apply(dFdu, adj_input)

        # TODO: First-order adjoint solution should be possible to obtain from the earlier adjoint computations.
        adj_sol = backend.Function(V)
        # Solve the (first order) adjoint equation
        compat.linalg_solve(dFdu, adj_sol.vector(), adj_input, **self.kwargs)

        # Second-order adjoint (soa) solution
        adj_sol2 = backend.Function(V)

        # Start piecing together the rhs of the soa equation
        b = hessian_input.copy()
        b_form = d2Fdu2
        b_vector = 0.0

        for _, bo in relevant_dependencies:
            c = bo.output
            c_rep = replaced_coeffs.get(c, c)
            tlm_input = bo.tlm_value

            if (c == self.func and not self.linear) or tlm_input is None:
                continue

            if not isinstance(c, backend.DirichletBC):
                b_form += backend.derivative(dFdu_form, c_rep, tlm_input)

        b_form = ufl.algorithms.expand_derivatives(b_form)
        if len(b_form.integrals()) > 0:
            b_form = backend.adjoint(b_form)
            b -= compat.assemble_adjoint_value(backend.action(b_form, adj_sol))
        b += b_vector
        b_copy = b.copy()

        for bc in bcs:
            bc.apply(dFdu, b)

        # Solve the soa equation
        compat.linalg_solve(dFdu, adj_sol2.vector(), b, **self.kwargs)

        adj_sol2_bdy = compat.function_from_vector(V, b_copy - compat.assemble_adjoint_value(
            backend.action(dFdu_form, adj_sol2)))

        r = {}
        r["adj_sol2"] = adj_sol2
        r["adj_sol2_bdy"] = adj_sol2_bdy
        r["form"] = F_form
        r["adj_sol"] = adj_sol
        r["replaced_coeffs"] = replaced_coeffs
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
        replaced_coeffs = prepared["replaced_coeffs"]
        fwd_block_variable = self.get_outputs()[0]
        tlm_output = fwd_block_variable.tlm_value

        c_rep = replaced_coeffs.get(c, c)

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
            c2_rep = replaced_coeffs.get(c2, c2)

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

    def prepare_recompute_component(self, inputs, relevant_outputs):
        func = backend.Function(self.get_outputs()[0].output.function_space())
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

        lhs = ufl.replace(self.lhs, replace_lhs_coeffs)

        rhs = 0
        if self.linear:
            rhs = ufl.replace(self.rhs, replace_rhs_coeffs)

        return lhs == rhs, func, bcs

    def recompute_component(self, inputs, block_variable, idx, prepared):
        eq = prepared[0]
        func = prepared[1]
        bcs = prepared[2]

        backend.solve(eq, func, bcs, **self.forward_kwargs)
        return func


class Form(object):
    def __init__(self, form, transpose=False):
        self.form = form
        self.rank = len(form.arguments())
        self.transpose = transpose
        self._data = None

        # Boundary conditions
        self.bcs = None
        self.bc_rows = None
        self.sol_var = None
        self.bc_type = 0

    def derivative(self, coefficient, argument=None, function_space=None):
        dc = argument
        if dc is None:
            if isinstance(coefficient, backend.Constant):
                dc = backend.Constant(1)
            elif isinstance(coefficient, backend.Expression):
                dc = backend.TrialFunction(function_space)

        diff_form = ufl.algorithms.expand_derivatives(backend.derivative(self.form, coefficient, dc))
        ret = Form(diff_form, transpose=self.transpose)
        ret.bcs = self.bcs
        ret.bc_rows = self.bc_rows
        ret.sol_var = self.sol_var

        # Unintuitive way of solving this problem.
        # TODO: Consider refactoring.
        if coefficient == self.sol_var:
            ret.bc_type = self.bc_type + 1
        else:
            ret.bc_type = self.bc_type + 2

        return ret

    def transpose(self):
        transpose = False if self.transpose else True
        return Form(self.form, transpose=transpose)

    def set_boundary_conditions(self, bcs, sol_var):
        self.bcs = []
        self.bc_rows = []
        self.sol_var = sol_var
        for bc in bcs:
            if isinstance(bc, backend.DirichletBC):
                bc = compat.create_bc(bc, homogenize=True)
            self.bcs.append(bc)

            # for key in bc.get_boundary_values():
            #     self.bc_rows.append(key)

    def apply_boundary_conditions(self, data):
        import numpy
        if self.bc_type >= 2:
            if self.rank >= 2:
                data.zero(numpy.array(self.bc_rows, dtype=numpy.intc))
            else:
                [bc.apply(data) for bc in self.bcs]
        else:
            [bc.apply(data) for bc in self.bcs]

    @property
    def data(self):
        return self.compute()

    def compute(self):
        if self._data is not None:
            return self._data

        if self.form.empty():
            return None

        data = backend.assemble(self.form)

        # Apply boundary conditions here!
        if self.bcs:
            self.apply_boundary_conditions(data)

        # Transpose if needed
        if self.transpose and self.rank >= 2:
            matrix_mat = backend.as_backend_type(data).mat()
            matrix_mat.transpose(matrix_mat)

        self._data = data
        return self._data

    def __mul__(self, other):
        if self.data is None:
            return 0

        if isinstance(other, Form):
            return self.data*other

        if isinstance(other, compat.MatrixType):
            if self.rank >= 2:
                return self.data*other
            else:
                # We (almost?) always want Matrix*Vector multiplication in this case.
                return other*self.data
        elif isinstance(other, compat.VectorType):
            if self.rank >= 2:
                return self.data*other
            else:
                return self.data.inner(other)

        # If it reaches this point I have done something wrong.
        return 0
