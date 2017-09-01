class Control(object):
    def __init__(self, control):
        self.control = control
        self.block_output = control.get_block_output()

    def data(self):
        return self.block_output.checkpoint

    def get_derivative(self, options={}):
        return self.control._ad_convert_type(self.block_output.adj_value, options=options)

    def __getattr__(self, item):
        return getattr(self.control, item)


