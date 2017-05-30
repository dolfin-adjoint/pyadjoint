from .tape import get_working_tape

# Type dependencies
from . import overloaded_type

class ReducedFunctional(object):
    """Class representing the reduced functional.

    A reduced functional maps a control value to the provided functional.
    It may also be used to compute the derivative of the functional with
    respect to the control.

    Args:
        functional (:obj:`OverloadedType`): An instance of an OverloadedType,
            usually :class:`AdjFloat`. This should be the return value of the
            functional you want to reduce.
        control (:obj:`OverloadedType`): An instance of an OverloadedType,
            which you want to map to the functional.

    """
    def __init__(self, functional, control):
        self.functional_block_output = functional.block_output 
        self.control = control
        self.tape = get_working_tape()

        for i, block in enumerate(self.tape.get_blocks()):
            if self.control.original_block_output in block.get_dependencies():
                self.block_idx = i
                break

    def derivative(self, options={}):
        """Returns the derivative of the functional w.r.t. the control.

        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the control,
        is computed and returned.
        
        Args:
            project (bool): If True returns the L^2 Riesz representation of the derivative. Otherwise the l^2 Riesz
                representation. Default is False.

        Returns:
            overloaded_type.OverloadedType: The derivative with respect to the control.
                Should be an instance of the same type as the control.

        """
        self.tape.reset_variables()
        self.functional_block_output.set_initial_adj_input(1.0)
        self.tape.evaluate(self.block_idx)

        return self.control.get_derivative(options=options)

    def __call__(self, value):
        """Computes the reduced functional with supplied control value.

        Args:
            value (:obj:`OverloadedType`): Should be an object of the same type
                as the control.

        Returns:
            :obj:`OverloadedType`: The computed value. Typically of instance
                of :class:`AdjFloat`.

        """
        self.control.adj_update_value(value)

        blocks = self.tape.get_blocks()
        for i in range(self.block_idx, len(blocks)):
            blocks[i].recompute()

        return self.functional_block_output.get_saved_output()





