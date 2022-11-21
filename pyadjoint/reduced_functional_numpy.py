from .reduced_functional import ReducedFunction, ReducedFunctional
from .tape import no_annotations
from .enlisting import Enlist
from .control import Control
from .adjfloat import AdjFloat

import numpy


class ReducedFunctionNumPy(ReducedFunction):
    """This class implements the reduced function for given function and
    controls based on NumPy data structures.

    This "NumPy version" of the ReducedFunction is created from
    an existing ReducedFunction object:
    rf_np = ReducedFunctionNumPy(rf = rf)
    """

    def __init__(self, reduced_function):
        if not isinstance(reduced_function, ReducedFunction):
            raise TypeError("reduced_function should be a ReducedFunction")

        self.rf = reduced_function
        self.output_controls = [Control(bv.output) for bv in self.outputs]

    def __getattr__(self, item):
        return getattr(self.rf, item)

    def get_rf_input(self, controls, m_array):
        """Converts a array to a list of OverloadedTypes using the given controls."""
        m_copies = [control.copy_data() for control in controls]
        return self.set_local_controls(controls, m_array, m_copies)

    def set_local(self, m_array, m):
        return self.set_local_controls(self.controls, m_array, m)

    def set_local_controls(self, controls, m_array, m):
        """Use the given numpy array to set the values of the given list of control
        values."""
        offset = 0
        for i, control in enumerate(controls):
            m[i], offset = control.assign_numpy(m[i], m_array, offset)
        return m

    def get_global(self, m):
        """Converts the given list of control values m to a single numpy array."""
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

    def get_outputs_array(self, values):
        m_global = []
        values = Enlist(values)
        for i, bv in enumerate(self.outputs):
            if values[i] is not None:
                m_global += bv.output._ad_to_list(values[i])
            else:
                m_global += [0] * bv.output._ad_dim()
        return numpy.array(m_global, dtype="d")

    @no_annotations
    def __call__(self, m_array):
        """An implementation of the reduced function evaluation
            that accepts the control values as an array of scalars.

        """
        m = self.get_rf_input(self.controls, m_array)
        output = self.rf(m)
        return self.get_outputs_array(output)

    @no_annotations
    def jac_action(self, h_array):
        """An implementation of the reduced function jac_action evaluation
        that accepts the controls as an array of scalars.
        """

        h = self.get_rf_input(self.controls, h_array)
        dJdm_h = self.rf.jac_action(h)
        return self.get_outputs_array(dJdm_h)

    @no_annotations
    def adj_jac_action(self, v_array):
        """An implementation of the reduced functional adjoint evaluation
        that returns the derivatives as an array of scalars.
        """

        v = self.get_rf_input(self.output_controls, v_array)
        v_dJdm = self.rf.adj_jac_action(v)
        v_dJdm = Enlist(v_dJdm)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that v_dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(v_dJdm[i])

        return numpy.array(m_global, dtype="d")

    @no_annotations
    def hess_action(self, h_array, v_array):
        """An implementation of the reduced function hessian action evaluation
        that accepts the controls as an array of scalars. If m_array is None,
        the Hessian action at the latest forward run is returned."""
        # Calling derivative is needed, see i.e. examples/stokes-shape-opt
        h = self.get_rf_input(self.controls, h_array)
        v = self.get_rf_input(self.output_controls, v_array)
        self.rf.adj_jac_action(v)

        v_hessian_h = self.rf.hess_action(h, v)
        v_hessian_h = Enlist(v_hessian_h)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(v_hessian_h[i])

        self.rf.tape.reset_variables()

        return numpy.array(m_global, dtype="d")

    def obj_to_array(self, obj):
        return self.get_global(obj)

    def get_controls(self, controls=None):
        if controls is None:
            controls = self.controls
        m = [p.tape_value() for p in controls]
        return self.obj_to_array(m)

    def set_controls(self, m_array):
        m = [p.tape_value() for p in self.controls]
        m = self.set_local_controls(self.controls, m_array, m)
        for control, m_i in zip(self.controls, m):
            control.update(m_i)
        return m


class ReducedFunctionalNumPy(ReducedFunctionNumPy):
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

    @no_annotations
    def derivative(self, m_array=None, forget=True, project=False):
        """ An implementation of the reduced functional derivative evaluation
            that accepts the controls as an array of scalars. If no control values are given,
            the result is derivative at the lastest forward run.
        """

        # In the case that the control values have changed since the last forward run,
        # we first need to rerun the forward model with the new controls to have the
        # correct forward solutions
        # TODO: No good way to check. Is it ok to always assume `m_array` is the same as used last in __call__?
        # if m_array is not None:
        #    self.__call__(m_array)
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
        """ An implementation of the reduced functional hessian action evaluation
            that accepts the controls as an array of scalars. If m_array is None,
            the Hessian action at the latest forward run is returned. """
        # Calling derivative is needed, see i.e. examples/stokes-shape-opt
        self.rf.derivative()
        h = self.get_rf_input(self.controls, m_dot_array)
        Hm = self.rf.hessian(h)
        Hm = Enlist(Hm)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(Hm[i])

        self.rf.tape.reset_variables()

        return numpy.array(m_global, dtype="d")


def set_local(coeffs, m_array):
    offset = 0
    for m in Enlist(coeffs):
        _, offset = m._ad_assign_numpy(m, m_array, offset)

    return coeffs


def gather(m):
    if isinstance(m, list):
        return list(map(gather, m))
    elif hasattr(m, "_ad_to_list"):
        return m._ad_to_list(m)
    else:
        return m  # Assume it is gathered already
