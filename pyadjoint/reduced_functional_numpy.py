from __future__ import print_function
from .reduced_functional import ReducedFunctional
from .tape import no_annotations, get_working_tape
from .enlisting import Enlist
from .control import Control
from .adjfloat import AdjFloat

import numpy


class ReducedFunctionalNumPy(ReducedFunctional):
    """This class implements the reduced functional for given functional and
    controls based on numpy data structures.

    This "NumPy version" of the pyadjoint.ReducedFunctional is created from
    an existing ReducedFunctional object:
    rf_np = ReducedFunctionalNumPy(rf = rf)
    """

    def __init__(self, functional, controls=None, tape=None):
        if isinstance(functional, AdjFloat):
            functional = ReducedFunctional(functional=functional,
                                           controls=controls,
                                           tape=tape)
        self.rf = functional

    def __getattr__(self, item):
        return getattr(self.rf, item)

    def __call__(self, m_array):
        """An implementation of the reduced functional evaluation
            that accepts the control values as an array of scalars

        """
        m_copies = [control.copy_data() for control in self.controls]
        return self.rf.__call__(self.set_local(m_copies, m_array))

    def set_local(self, m, m_array):
        offset = 0
        for i, control in enumerate(self.controls):
            _, offset = control.assign_numpy(m[i], m_array, offset)

        return m

    def get_global(self, m):
        m_global = []
        for i, v in enumerate(Enlist(m)):
            if isinstance(v, Control):
                # TODO: Consider if you want this design.
                m_global += v.fetch_numpy(v.control)
            elif hasattr(v, "_ad_to_list"):
                m_global += v._ad_to_list(v)
            else:
                m_global += self.controls[i].control._ad_to_list(v)
        return numpy.array(m_global, dtype="d")

    @no_annotations
    def derivative(self, m_array=None, forget=True, project=False):
        ''' An implementation of the reduced functional derivative evaluation
            that accepts the controls as an array of scalars. If no control values are given,
            the result is derivative at the lastest forward run.
        '''

        # In the case that the control values have changed since the last forward run,
        # we first need to rerun the forward model with the new controls to have the
        # correct forward solutions
        if m_array is not None:
            self.__call__(m_array)
        dJdm = self.rf.derivative()
        dJdm = Enlist(dJdm)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(dJdm[i])

        return numpy.array(m_global, dtype="d")

    @no_annotations
    def hessian(self, m_array, m_dot_array):
        ''' An implementation of the reduced functional hessian action evaluation
            that accepts the controls as an array of scalars. If m_array is None,
            the Hessian action at the latest forward run is returned. '''
        # TODO: Consider if we really need to run derivative here.
        self.derivative()
        m_copies = [control.copy_data() for control in self.controls]
        Hm = self.rf.hessian(self.set_local(m_copies, m_dot_array))
        Hm = Enlist(Hm)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(Hm[i])

        tape = get_working_tape()
        tape.reset_variables()

        return numpy.array(m_global, dtype="d")

    def obj_to_array(self, obj):
        return self.get_global(obj)

    def get_controls(self):
        m = [p.data() for p in self.controls]
        return self.obj_to_array(m)

    def set_controls(self, array):
        m = [p.data() for p in self.controls]
        return self.set_local(m, array)

    def pyopt_problem(self, constraints=None, bounds=None, name="Problem", ignore_model_errors=False):
        '''Return a pyopt problem class that can be used with the PyOpt package,
        http://www.pyopt.org/
        '''
        import pyOpt
        from .optimization import constraints

        constraints = optimization.constraints.canonicalise(constraints)

        def obj(x):
            ''' Evaluates the functional for the given controls values. '''

            fail = False
            if not ignore_model_errors:
                j = self(x)
            else:
                try:
                    j = self(x)
                except:
                    fail = True

            if constraints is not None:
                # Not sure how to do this in parallel, FIXME
                g = np.concatenate(constraints.function(x))
            else:
                g = [0]  # SNOPT fails if no constraints are given, hence add a dummy constraint

            return j, g, fail

        def grad(x, f, g):
            ''' Evaluates the gradient for the control values.
            f is the associated functional value and g are the values
            of the constraints. '''

            fail = False
            if not ignore_model_errors:
                dj = self.derivative(x, forget=False)
            else:
                try:
                    dj = self.derivative(x, forget=False)
                except:
                    fail = True

            if constraints is not None:
                gJac = np.concatenate([gather(c.jacobian(x)) for c in constraints])
            else:
                gJac = np.zeros(len(x))  # SNOPT fails if no constraints are given, hence add a dummy constraint

            info("j = %f\t\t|dJ| = %f" % (f[0], np.linalg.norm(dj)))
            return np.array([dj]), gJac, fail

        # Instantiate the optimization problem
        opt_prob = pyOpt.Optimization(name, obj)
        opt_prob.addObj('J')

        # Compute bounds
        m = self.get_controls()
        n = len(m)

        if bounds is not None:
            bounds_arr = [None, None]
            for i in range(2):
                if isinstance(bounds[i], float) or isinstance(bounds[i], int):
                    bounds_arr[i] = np.ones(n) * bounds[i]
                else:
                    bounds_arr[i] = np.array(bounds[i])
            lb, ub = bounds_arr

        else:
            mx = np.finfo(np.double).max
            ub = mx * np.ones(n)

            mn = np.finfo(np.double).min
            lb = mn * np.ones(n)

        # Add controls
        opt_prob.addVarGroup("variables", n, type='c', value=m, lower=lb, upper=ub)

        # Add constraints
        if constraints is not None:
            for i, c in enumerate(constraints):
                if isinstance(c, optimization.constraints.EqualityConstraint):
                    opt_prob.addConGroup(str(i) + 'th constraint', c._get_constraint_dim(), type='e', equal=0.0)
                elif isinstance(c, optimization.constraints.InequalityConstraint):
                    opt_prob.addConGroup(str(i) + 'th constraint', c._get_constraint_dim(), type='i', lower=0.0, upper=np.inf)

        return opt_prob, grad


def set_local(coeffs, m_array):
    offset = 0
    for m in Enlist(coeffs):
        _, offset = m._ad_assign_numpy(m, m_array, offset)

    return coeffs


def gather(m):
    if isinstance(m, list):
        return list(map(gather, m))
    else:
        return m._ad_to_list(m)
