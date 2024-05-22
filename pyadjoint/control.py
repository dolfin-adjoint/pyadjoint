from .overloaded_type import OverloadedType, create_overloaded_object
import logging


class Control(object):
    """Defines a control variable from an OverloadedType.

    The control object references a specific node on the Tape.
    For mutable OverloadedType instances the Control only represents
    the value at the time of initialization.

    Example:
        Given a mutable OverloadedType instance u.

        >>> u = MutableFloat(1.0)
        >>> float(u)
        1.0
        >>> c1 = Control(u)
        >>> u.add_in_place(2.0)
        >>> c2 = Control(u)
        >>> float(u)
        3.0
        >>> c1.data()
        1.0
        >>> c2.data()
        3.0

        Now c1 represents the node prior to the add_in_place Block,
        while c2 represents the node after the add_in_place Block.
        Creating a `ReducedFunctional` with c2 as Control results in
        a reduced problem without the add_in_place Block, while a ReducedFunctional
        with c1 as Control results in a forward model including the add_in_place.

    Args:
        control (OverloadedType): The OverloadedType instance to define this control from.

    """
    def __init__(self, control):
        self.control = control
        self.block_variable = control.block_variable

    def data(self):
        return self.block_variable.checkpoint

    def tape_value(self):
        return create_overloaded_object(self.block_variable.saved_output)

    def get_derivative(self, options={}):
        if self.block_variable.adj_value is None:
            logging.warning("Adjoint value is None, is the functional independent of the control variable?")
            return self.control._ad_convert_type(0., options=options)
        return self.control._ad_convert_type(self.block_variable.adj_value, options=options)

    def get_hessian(self, options={}):
        if self.block_variable.adj_value is None:
            logging.warning("Hessian value is None, is the functional independent of the control variable?")
            return self.control._ad_convert_type(0., options=options)
        return self.control._ad_convert_type(self.block_variable.hessian_value, options=options)

    def update(self, value):
        # In the future we might want to call a static method
        # for converting a value to the correct type.
        # As this might depend on the OverloadedType control.
        if isinstance(value, OverloadedType):
            self.block_variable.checkpoint = value._ad_create_checkpoint()
        else:
            self.block_variable.checkpoint = value

    def update_numpy(self, value, offset):
        self.block_variable.checkpoint, offset =\
            self.assign_numpy(self.block_variable.checkpoint, value, offset)
        return offset

    def assign_numpy(self, dst, src, offset):
        # This returns dst and offset. dst needs to be returned in case
        # it is immutable. Because then we cannot assign in-place, but need
        # to create a new object.
        return self.control._ad_assign_numpy(dst, src, offset)

    def fetch_numpy(self, value):
        return self.control._ad_to_list(value)

    def copy_data(self):
        return self.control._ad_copy()

    @property
    def adj_value(self):
        return self.block_variable.adj_value

    @adj_value.setter
    def adj_value(self, value):
        self.block_variable.adj_value = value

    @property
    def tlm_value(self):
        return self.block_variable.tlm_value

    @tlm_value.setter
    def tlm_value(self, value):
        self.block_variable.tlm_value = value

    @property
    def hessian_value(self):
        return self.block_variable.hessian_value

    @hessian_value.setter
    def hessian_value(self, value):
        self.block_variable.hessian_value = value

    def __getattr__(self, item):
        return getattr(self.control, item)

    def mark_as_control(self):
        self.block_variable.is_control = True

    def unmark_as_control(self):
        self.block_variable.is_control = False
