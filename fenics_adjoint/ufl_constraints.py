import backend
import ufl.algorithms
import types
from pyadjoint.optimization.constraints import Constraint, EqualityConstraint, InequalityConstraint

def as_vec(x):
    if backend.__name__ in ["dolfin", "fenics"]:
        if isinstance(x, backend.Function):
            return x.vector().array()
        else:
            return x.array()
    elif backend.__name__ == "firedrake":
        return x
    else:
        raise NotImplementedError("Unknown backend")

class UFLConstraint(Constraint):
    def __init__(self, form, control):

        if not isinstance(control.control, backend.Function):
            raise NotImplementedError("Only implemented for Function controls")

        args = ufl.algorithms.extract_arguments(form)
        if len(args) != 1:
            raise ValueError("Must be a rank-one form")
        self.tV = args[0].function_space()

        self.form = form
        self.control = control

        self.V = control.control.function_space()

        self.trial1 = backend.TrialFunction(self.V)
        self.trial2 = backend.TrialFunction(self.V)

        self.dform = backend.derivative(form, control.control, self.trial1)

        if len(ufl.algorithms.extract_arguments(ufl.algorithms.expand_derivatives(self.dform))) == 0:
            raise ValueError("Form must depend on control")
        self.adform = backend.adjoint(self.dform)
        self.hess = backend.derivative(form, control.control, self.trial2)

    def update_control(self, m):
        # self.control.update(m) crashes; quite counterintuitive
        if backend.__name__ in ["dolfin", "fenics"]:
            self.control.control.vector().set_local(m)
        else:
            raise NotImplementedError("Not implemented for firedrake yet")

    def function(self, m):
        self.update_control(m)
        b = backend.assemble(self.form)
        return as_vec(b)

    def jacobian(self, m):
        self.update_control(m)
        J = backend.assemble(self.dform)

        # We need to make the matrix dense, then extract it row-by-row, then
        # return the columns as a list.
        if backend.__name__ in ["dolfin", "fenics"]:
            out = []
            for i in range(J.size(0)):
                (cols, vals) = J.getrow(i)
                v = types.Function(self.V)
                v.vector().set_local(vals) # is there a better way to do this?
                out.append(v)
        else:
            raise NotImplementedError("Not implemented for firedrake yet")
        return out

    def jacobian_action(self, m, dm, result):
        """Computes the Jacobian action of c(m) in direction dm and stores the result in result. """

        self.update_control(m)
        if isinstance(result, backend.Function):
            result.vector().assign(backend.assemble(backend.action(self.dform, dm)))
        else:
            raise NotImplementedError("Do I need to untangle all controls?")

    def jacobian_adjoint_action(self, m, dp, result):
        """Computes the Jacobian adjoint action of c(m) in direction dp and stores the result in result. """

        self.update_control(m)
        if isinstance(result, backend.Function):
            result.vector().assign(backend.assemble(backend.action(self.adform, dp)))
        else:
            raise NotImplementedError("Do I need to untangle all controls?")

    def hessian_action(self, m, dm, dp, result):
        """Computes the Hessian action of c(m) in direction dm and dp and stores the result in result. """

        self.update_control(m)
        H = backend.replace(self.hess, {self.trial1: dm, self.trial2: dp})
        if isinstance(result, backend.Function):
            result.vector().assign(backend.assemble(H))
        else:
            raise NotImplementedError("Do I need to untangle all controls?")

    def output_workspace(self):
        """Return an object like the output of c(m) for calculations."""

        return as_vec(backend.assemble(self.form))

    def _get_constraint_dim(self):
        """Returns the number of constraint components."""
        return self.tV.dim()

class UFLEqualityConstraint(UFLConstraint, EqualityConstraint):
    pass

class UFLInequalityConstraint(UFLConstraint, InequalityConstraint):
    pass
