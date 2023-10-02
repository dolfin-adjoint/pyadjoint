from .tape import no_annotations, get_working_tape


class BlockVariable(object):
    """References a block output variable.

    """

    def __init__(self, output):
        self.output = output
        self.adj_value = None
        self.tlm_value = None
        self.hessian_value = None
        self._checkpoint = None
        self.is_control = False
        self.floating_type = False
        # Helper flag for use during tape traversals.
        self.marked_in_path = False
        # By default assume the variable is created externally to the tape.
        self.creation_timestep = -1
        # The timestep during which this variable was last used as an input.
        self.last_use = -1

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

    def reset_variables(self, types):
        if "adjoint" in types:
            self.adj_value = None

        if "hessian" in types:
            self.hessian_value = None

        if "tlm" in types:
            self.tlm_value = None

    @no_annotations
    def save_output(self, overwrite=True):
        if overwrite or self.checkpoint is None:
            self._checkpoint = self.output._ad_create_checkpoint()

    @property
    def saved_output(self):
        if self.checkpoint is not None:
            return self.output._ad_restore_at_checkpoint(self.checkpoint)
        else:
            return self.output

    def will_add_as_dependency(self):
        tape = get_working_tape()
        if tape._time_dependent:
            overwrite = self.output._ad_will_add_as_dependency()
            overwrite = False if overwrite is None else overwrite
            
            if self.last_use < tape.latest_checkpoint:
                self.save_output(overwrite=overwrite)
            tape.add_to_checkpointable_state(self, self.last_use)
            self.last_use = tape.latest_timestep
        else:
            overwrite = self.output._ad_will_add_as_dependency()
            overwrite = False if overwrite is None else overwrite
            self.save_output(overwrite=overwrite)

    def will_add_as_output(self):
        if get_working_tape()._time_dependent:
            tape = get_working_tape()
            self.creation_timestep = tape.latest_timestep
            self.last_use = self.creation_timestep
            overwrite = self.output._ad_will_add_as_output()
            overwrite = True if overwrite is None else overwrite
            if overwrite:
                self._checkpoint = None
            if tape._eagerly_checkpoint_outputs:
                self.save_output()
        else:
            overwrite = self.output._ad_will_add_as_output()
            overwrite = True if overwrite is None else overwrite
            self.save_output(overwrite=overwrite)

    def __str__(self):
        return str(self.output)

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value):
        if self.is_control:
            return
        self._checkpoint = value
