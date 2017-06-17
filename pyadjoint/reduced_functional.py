from .tape import get_working_tape, pause_annotation, continue_annotation
from .drivers import compute_gradient
from .overloaded_type import OverloadedType


class ReducedFunctional(object):
    """Class representing the reduced functional.

    A reduced functional maps a control value to the provided functional.
    It may also be used to compute the derivative of the functional with
    respect to the control.

    Args:
        functional (:obj:`OverloadedType`): An instance of an OverloadedType,
            usually :class:`AdjFloat`. This should be the return value of the
            functional you want to reduce.
        controls (OverloadedType): An instance of an OverloadedType,
            which you want to map to the functional. You may also supply a list
            of such instances if you have multiple controls.

    """
    def __init__(self, functional, controls):
        self.functional = functional
        self.tape = get_working_tape()

        if isinstance(controls, OverloadedType):
            self.controls = [controls]
        else:
            self.controls = controls

        for i, block in enumerate(self.tape.get_blocks()):
            for control in self.controls:
                if control.original_block_output in block.get_dependencies():
                    self.block_idx = i
                    return

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
        derivatives = compute_gradient(self.functional, self.controls, options=options, tape=self.tape)
        if len(derivatives) == 1:
            return derivatives[0]
        else:
            return derivatives

    def __call__(self, values):
        """Computes the reduced functional with supplied control value.

        Args:
            values (:obj:`OverloadedType`): Should be an object of the same type
                as the control.

        Returns:
            :obj:`OverloadedType`: The computed value. Typically of instance
                of :class:`AdjFloat`.

        """
        if isinstance(values, OverloadedType):
            self.controls[0].adj_update_value(values)
        else:
            for i, value in enumerate(values):
                self.controls[i].adj_update_value(value)

        blocks = self.tape.get_blocks()
        pause_annotation()
        for i in range(self.block_idx, len(blocks)):
            blocks[i].recompute()
        continue_annotation()

        return self.functional.block_output.get_saved_output()





