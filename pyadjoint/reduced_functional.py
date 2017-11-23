from .tape import get_working_tape, stop_annotating, Tape, no_annotations
from .drivers import compute_gradient
from .overloaded_type import OverloadedType
from .control import Control
from .enlisting import Enlist

class ReducedFunctional(object):
    """Class representing the reduced functional.

    A reduced functional maps a control value to the provided functional.
    It may also be used to compute the derivative of the functional with
    respect to the control.

    Args:
        functional (:obj:`OverloadedType`): An instance of an OverloadedType,
            usually :class:`AdjFloat`. This should be the return value of the
            functional you want to reduce.
        controls (list[Control]): A list of Control instances, which you want
            to map to the functional. It is also possible to supply a single Control
            instance instead of a list.

    """
    def __init__(self, functional, controls, tape=None):
        self.functional = functional
        self.tape = get_working_tape() if tape is None else tape
        self.controls = Enlist(controls)

    def derivative(self, options={}):
        """Returns the derivative of the functional w.r.t. the control.

        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the control,
        is computed and returned.
        
        Args:
            options (dict): A dictionary of options. To find a list of available options
                have a look at the specific control type.

        Returns:
            OverloadedType: The derivative with respect to the control.
                Should be an instance of the same type as the control.

        """
        derivatives = compute_gradient(self.functional,
                                       self.controls,
                                       options=options,
                                       tape=self.tape)
        return self.controls.delist(derivatives)

    @no_annotations
    def __call__(self, values):
        """Computes the reduced functional with supplied control value.

        Args:
            values ([OverloadedType]): If you have multiple controls this should be a list of
                new values for each control in the order you listed the controls to the constructor.
                If you have a single control it can either be a list or a single object.
                Each new value should have the same type as the corresponding control.

        Returns:
            :obj:`OverloadedType`: The computed value. Typically of instance
                of :class:`AdjFloat`.

        """
        values = Enlist(values)
        if len(values) != len(self.controls):
            raise ValueError("values should be a list of same length as controls.")

        for i, value in enumerate(values):
            self.controls[i].update(value)

        blocks = self.tape.get_blocks()
        with self.marked_controls():
            with stop_annotating():
                for i in range(len(blocks)):
                    blocks[i].recompute()

        return self.functional.block_output.checkpoint

    def optimize(self):
        self.tape.optimize(
            controls=self.controls,
            functionals=[self.functional]
        )

    def marked_controls(self):
        return marked_controls(self)


class marked_controls(object):
    def __init__(self, rf):
        self.rf = rf

    def __enter__(self):
        for control in self.rf.controls:
            control.mark_as_control()

    def __exit__(self, *args):
        for control in self.rf.controls:
            control.unmark_as_control()









