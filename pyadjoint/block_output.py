class BlockOutput(object):
    """References a block output variable.

    """

    def __init__(self, output):
        self.output = output
        self.adj_value = None
        self.tlm_value = None
        self.hessian_value = None
        self.checkpoint = None

    def add_adj_output(self, val):
        if self.adj_value is None:
            self.adj_value = val
        else:
            self.adj_value += val

    def add_tlm_output(self, val):
        if self.tlm_value is None:
            self.tlm_value = val
        else:
            self.tlm_value += val

    def add_hessian_output(self, val):
        if self.hessian_value is None:
            self.hessian_value = val
        else:
            self.hessian_value += val

    def get_adj_output(self):
        return self.adj_value

    def set_initial_adj_input(self, value):
        self.adj_value = value

    def set_initial_tlm_input(self, value):
        self.tlm_value = value

    def reset_variables(self):
        self.adj_value = None

    # TODO: Make this just an attribute. Extend with Property if needed later.
    def get_output(self):
        return self.output

    def save_output(self, overwrite=True):
        if overwrite or not self.checkpoint:
            self.checkpoint = self.output._ad_create_checkpoint()

    def get_saved_output(self):
        if self.checkpoint is not None:
            return self.output._ad_restore_at_checkpoint(self.checkpoint)
        else:
            return self.output

    def __str__(self):
        return str(self.output)

