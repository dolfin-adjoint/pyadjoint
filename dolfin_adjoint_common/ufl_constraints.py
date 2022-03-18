import backend
import ufl
import ufl.algorithms
import numpy

from pyadjoint.enlisting import Enlist
from pyadjoint.optimization.constraints import Constraint, EqualityConstraint, InequalityConstraint


if backend.__name__ in ["dolfin", "fenics"]:
    import fenics_adjoint.types as backend_types
else:
    backend_types = backend


def as_vec(x):
    if backend.__name__ in ["dolfin", "fenics"]:
        if isinstance(x, backend.Function):
            out = x.vector().get_local()
        else:
            out = x.get_local()

        if len(out) == 1:
            out = out[0]
        return backend_types.Constant(out)
    elif backend.__name__ == "firedrake":
        with x.dat.vec_ro as vec:
            copy = numpy.array(vec)
        if len(copy) == 1:
            copy = copy[0]
        return backend_types.Constant(copy)
    else:
        raise NotImplementedError("Unknown backend")


class UFLConstraint(Constraint):
    """
    Easily implement scalar constraints using UFL.

    The form must be a 0-form that depends on a Function control.
    """

    def __init__(self, form, controls):

        controls = Enlist(controls)
        for control in controls:
            if not isinstance(control.control, backend.Function):
                raise NotImplementedError("Only implemented for Function controls")

        args = ufl.algorithms.extract_arguments(form)
        if len(args) != 0:
            raise ValueError("Must be a rank-zero form, i.e. a functional")

        replace_dict = {}
        functions = []
        for control in controls:
            u = control.control
            V = u.function_space()
            u_placeholder = backend_types.Function(V)

            functions.append(u_placeholder)
            replace_dict[u] = u_placeholder

        self.form = ufl.replace(form, replace_dict)
        self.functions = functions
        self.dforms = [ufl.algorithms.expand_derivatives(backend.derivative(self.form, u)) for u in functions]
        self.hess = [[ufl.algorithms.expand_derivatives(backend.derivative(dform, u)) for dform in self.dforms] for u in functions]
        # TODO: Add check for at least one control dependency.

    def update_control(self, m):
        m = Enlist(m)
        global_numpy = len(m) != len(self.functions)

        assert not global_numpy or len(m) == 1, "Assumption of a single global numpy array is wrong."

        offset = 0
        for i, ui in enumerate(self.functions):
            mi = m[0 if global_numpy else i]
            if isinstance(mi, backend.Function):
                ui.assign(mi)
            else:
                _, offset = ui._ad_assign_numpy(ui, mi, offset)

    def function(self, m):
        self.update_control(m)
        b = backend.assemble(self.form)
        return backend_types.Constant(b)

    def jacobian(self, m):
        m = Enlist(m)
        self.update_control(m)
        out = []
        for mi, dform in zip(self.functions, self.dforms):
            if dform.empty():
                out.append(backend.assemble(backend.inner(backend.Constant(numpy.zeros(mi.ufl_shape)), backend.TestFunction(mi.function_space()))*backend.dx))
            else:
                out.append(backend.assemble(dform))
        return out

    def jacobian_action(self, m, dm, result):
        """Computes the Jacobian action of c(m) in direction dm and stores the result in result. """
        m = Enlist(m)
        dm = Enlist(dm)
        self.update_control(m)

        form = sum([backend.action(dform, dmi) for dform, dmi in zip(self.dforms, dm) if not dform.empty()])
        result.assign(backend.assemble(form))

    def jacobian_adjoint_action(self, m, dp, result):
        """Computes the Jacobian adjoint action of c(m) in direction dp and stores the result in result. """
        m = Enlist(m)
        result = Enlist(result)
        self.update_control(m)

        for dform, res in zip(self.dforms, result):
            if isinstance(res, backend.Function):
                asm = backend.assemble(dp * dform) if not dform.empty() else None
                if backend.__name__ in ["dolfin", "fenics"]:
                    res.vector().zero()
                    if asm is not None:
                        res.vector().axpy(1.0, asm)
                else:
                    # TODO: Is it safe to assuming default is zero?
                    if asm is not None:
                        res.assign(asm)
            else:
                raise NotImplementedError(f"Unsupported result type in constraint jacobian adjoint action: {type(res)}.")

    def hessian_action(self, m, dm, dp, result):
        """Computes the Hessian action of c(m) in direction dm and dp and stores the result in result. """
        m = Enlist(m)
        dm = Enlist(dm)
        self.update_control(m)

        for hess, res in zip(self.hess, result):
            H = sum([dp * backend.action(hessi, dmi) for hessi, dmi in zip(hess, dm) if not hessi.empty()])
            empty = isinstance(H, (int, float)) or H.empty()
            if isinstance(res, backend.Function):
                if backend.__name__ in ["dolfin", "fenics"]:
                    if empty:
                        res.vector().zero()
                    else:
                        res.vector().zero()
                        res.vector().axpy(1.0, backend.assemble(H))
                else:
                    if empty:
                        res.assign(0)
                    else:
                        res.assign(backend.assemble(H))
            else:
                raise NotImplementedError(f"Unsupported result type in constraint hessian action: {type(res)}.")

    def output_workspace(self):
        """Return an object like the output of c(m) for calculations."""

        return backend_types.Constant(backend.assemble(self.form))

    def _get_constraint_dim(self):
        """Returns the number of constraint components."""
        return 1


class UFLEqualityConstraint(UFLConstraint, EqualityConstraint):
    pass


class UFLInequalityConstraint(UFLConstraint, InequalityConstraint):
    pass
