from .tape import get_working_tape
from .block_output import BlockOutput


class OverloadedType(object):
    """Base class for OverloadedType types.

    The purpose of each OverloadedType is to extend a type such that
    it can be referenced by blocks as well as overload basic mathematical
    operations such as __mul__, __add__, where they are needed.

    Abstract methods:
        :func:`adj_update_value`

    """
    def __init__(self, *args, **kwargs):
        tape = kwargs.pop("tape", None)

        if tape:
            self.tape = tape
        else:
            self.tape = get_working_tape()

        self.original_block_output = self.create_block_output()

    def create_block_output(self):
        block_output = BlockOutput(self)
        self.set_block_output(block_output)
        return block_output

    def set_block_output(self, block_output):
        self.block_output = block_output

    def get_block_output(self):
        return self.block_output

    def get_adj_output(self):
        return self.original_block_output.get_adj_output()

    def set_initial_adj_input(self, value):
        self.block_output.set_initial_adj_input(value)

    def reset_variables(self):
        self.original_block_output.reset_variables()

    def get_derivative(self, options={}):
        # TODO: Decide on naming here.
        # Basically the method should implement a way to convert
        # the adj_output to the same type as `self`.
        raise NotImplementedError

    def _ad_create_checkpoint(self):
        """This method must be overridden.
        
        Should implement a way to create a checkpoint for the overloaded object.
        The checkpoint should be returned and possible to restore from in the
        corresponding _ad_restore_at_checkpoint method.

        Returns:
            :obj:`object`: A checkpoint. Could be of any type, but must be possible
                to restore an object from that point.

        """
        raise NotImplementedError

    def _ad_restore_at_checkpoint(self, checkpoint):
        """This method must be overridden.

        Should implement a way to restore the object at supplied checkpoint.
        The checkpoint is created from the _ad_create_checkpoint method.

        Returns:
            :obj:`OverloadedType`: The object with same state as at the supplied checkpoint.

        """
        raise NotImplementedError

    def adj_update_value(self, value):
        """This method must be overridden.

        The method should implement a routine for assigning a new value
        to the overloaded object.

        Args:
            value (:obj:`object`): Should be an instance of the OverloadedType.

        """
        raise NotImplementedError

    def _ad_mul(self, other):
        """This method must be overridden.

        The method should implement a routine for multiplying the overloaded object
        with another object, and return an object of the same type as `self`.

        Args:
            other (:obj:`object`): The object to be multiplied with this.
                Should at the very least accept :obj:`float` and :obj:`integer` objects.

        Returns:
            :obj:`OverloadedType`: The product of the two objects represented as
                an instance of the same subclass of :class:`OverloadedType` as the type
                of `self`.

        """
        raise NotImplementedError

    def _ad_add(self, other):
        """This method must be overridden.

        The method should implement a routine for adding the overloaded object
        with another object, and return an object of the same type as `self`.

        Args:
            other (:obj:`object`): The object to be added with this.
                Should at the very least accept objects of the same type as `self`.

        Returns:
            :obj:`OverloadedType`: The sum of the two objects represented as
                an instance of the same subclass of :class:`OverloadedType` as the type
                of `self`.

        """
        raise NotImplementedError

    def _ad_dot(self, other):
        """This method must be overridden.

        The method should implement a routine for computing the dot product of
        the overloaded object with another object of the same type, and return
        a :obj:`float`.

        Args:
            other (:obj:`OverloadedType`): The object to compute the dot product with.
                Should be of the same type as `self`.

        Returns:
            :obj:`float`: The dot product of the two objects.

        """
        raise NotImplementedError
