from .overloaded_type import OverloadedType


class Control(object):
    def __init__(self, control):
        self.control = control
        self.block_output = control.get_block_output()

    def data(self):
        return self.block_output.checkpoint

    def get_derivative(self, options={}):
        return self.control._ad_convert_type(self.block_output.adj_value, options=options)

    def update(self, value):
        # In the future we might want to call a static method
        # for converting a value to the correct type.
        # As this might depend on the OverloadedType control.
        if isinstance(value, OverloadedType):
            self.block_output.checkpoint = value._ad_create_checkpoint()
        else:
            self.block_output.checkpoint = value

    def __getattr__(self, item):
        return getattr(self.control, item)


