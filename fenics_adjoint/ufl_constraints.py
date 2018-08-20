import backend
import ufl.algorithms
import numpy

if backend.__name__ in ["dolfin", "fenics"]:
    import fenics_adjoint.types as backend_types
elif backend.__name__ == "firedrake":
    import firedrake_adjoint.types as backend_types
else:
    raise NotImplementedError("Unknown backend")

from pyadjoint.optimization.constraints import Constraint, EqualityConstraint, InequalityConstraint

def as_vec(x):
    if backend.__name__ in ["dolfin", "fenics"]:
        if isinstance(x, backend.Function):
            return x.vector().get_local()
        else:
            return x.get_local()
    elif backend.__name__ == "firedrake":
        with x.dat.vec_ro as vec:
            copy = numpy.array(vec)
        return copy
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

        u = control.control
        self.V = u.function_space()
        # We want to make a copy of the control purely for use
        # in the constraint, so that our writing it isn't
        # bothering anyone else
        self.u = backend.Function(self.V)
        self.form = backend.replace(form, {u: self.u})


        self.trial1 = backend.TrialFunction(self.V)

        self.dform = backend.derivative(self.form, self.u, self.trial1)

        if len(ufl.algorithms.extract_arguments(ufl.algorithms.expand_derivatives(self.dform))) == 0:
            raise ValueError("Form must depend on control")
        self.adform = backend.adjoint(self.dform)
        self.hess = backend.derivative(self.dform, self.u)
        (_, _, self.trial2) = self.hess.arguments()

    def update_control(self, m):
        if backend.__name__ in ["dolfin", "fenics"]:
            self.u.vector().set_local(m)
        else:
            with self.u.dat.vec_wo as x:
                x[:] = m

    def function(self, m):
        self.update_control(m)
        b = backend.assemble(self.form)
        return as_vec(b)

    def jacobian(self, m):
        self.update_control(m)

        # We need to make the matrix dense, then extract it row-by-row, then
        # return the columns as a list.
        if backend.__name__ in ["dolfin", "fenics"]:
            J = backend.assemble(self.dform)
            out = []
            for i in range(J.size(0)):
                (cols, vals) = J.getrow(i)
                v = backend_types.Function(self.V)
                v.vector().set_local(vals) # is there a better way to do this?
                out.append(v)
        else:
            out = []
            J = backend.assemble(self.dform, mat_type="aij")
            J.force_evaluation()
            M = J.petscmat
            if M.type == "python":
                if M.size[0] != 1:
                    # I don't know what data structure firedrake uses here yet, because they haven't coded it
                    raise NotImplementedError("This case isn't supported by PyOP2 at time of writing")
                else:
                    ctx = M.getPythonContext()
                    v = backend_types.Function(self.V)
                    with v.dat.vec as x, ctx.dat.vec_ro as y:
                        y.copy(x)
                    out.append(v)
            else:
                raise NotImplementedError("Not encountered this case yet, patches welcome")
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
        if backend.__name__ in ["dolfin", "fenics"]:
            return self.tV.dim()
        elif backend.__name__ == "firedrake":
            return self.tV.dim
        else:
            raise NotImplementedError("Unknown backend")

class UFLEqualityConstraint(UFLConstraint, EqualityConstraint):
    pass

class UFLInequalityConstraint(UFLConstraint, InequalityConstraint):
    pass
