from __future__ import print_function
from .reduced_functional import AbstractReducedFunctional, ReducedFunctional
from .tape import no_annotations, get_working_tape
from .enlisting import Enlist
from .control import Control
from .adjfloat import AdjFloat

import numpy


class ReducedFunctionalNumPy(AbstractReducedFunctional):
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

    @property
    def controls(self) -> list[Control]:
        return self.rf._controls

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
            m[i], offset = control.assign_numpy(m[i], m_array, offset)

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
    def derivative(self, adj_input=1.0, apply_riesz=True):

        if not apply_riesz:
            raise ValueError(
                "ReducedFunctionalNumpy only returns primal gradients."
            )

        dJdm = self.rf.derivative(adj_input=adj_input,
                                  apply_riesz=apply_riesz)
        dJdm = Enlist(dJdm)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(dJdm[i])

        return numpy.array(m_global, dtype="d")

    @no_annotations
    def hessian(self, m_dot_array, apply_riesz=True):

        if not apply_riesz:
            raise ValueError(
                "ReducedFunctionalNumpy only returns primal gradients."
            )

        # Calling derivative is needed, see i.e. examples/stokes-shape-opt
        self.derivative()
        m_copies = [control.copy_data() for control in self.controls]
        Hm = self.rf.hessian(self.set_local(m_copies, m_dot_array),
                             apply_riesz=apply_riesz)
        Hm = Enlist(Hm)

        m_global = []
        for i, control in enumerate(self.controls):
            # This is a little ugly, but we need to go through the control to get to the OverloadedType.
            # There is no guarantee that dJdm[i] is an OverloadedType and not a backend type.
            m_global += control.fetch_numpy(Hm[i])

        tape = get_working_tape()
        tape.reset_variables()

        return numpy.array(m_global, dtype="d")

    @no_annotations
    def tlm(self, m_dot_array):
        m_copies = [control.copy_data() for control in self.controls]
        tm = self.rf.tlm(self.set_local(m_copies, m_dot_array))
        tm_global = self.functional._ad_to_list(tm)

        tape = get_working_tape()
        tape.reset_variables()

        return numpy.array(tm_global, dtype="d")

    def obj_to_array(self, obj):
        return self.get_global(obj)

    def get_controls(self):
        m = [p.tape_value() for p in self.controls]
        return self.obj_to_array(m)

    def set_controls(self, array):
        m = [p.tape_value() for p in self.controls]
        m = self.set_local(m, array)
        for control, m_i in zip(self.controls, m):
            control.update(m_i)
        return m


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
