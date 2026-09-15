import operator

from .tape import no_annotations, get_working_tape


def _accumulate(current, val):
    """Add `val` into `current` and return the accumulated value.

    The caller must rebind to the return value rather than relying on
    `_ad_iadd` to mutate in place: an immutable value type cannot accumulate in
    place, so its `_ad_iadd` returns a new object. `AdjFloat` subclasses
    `float`, so this is the common case, not a corner one. Mutable types return
    `self`, for which the rebinding is a no-op.

    `_ad_iadd` returning `None` is taken to mean "mutated in place, keep what
    you had". Overrides written against the pre-2026 contract, which documented
    `Returns: None`, would otherwise wipe the accumulated value and silently
    drop every contribution so far.

    A block is not obliged to return an `OverloadedType` at all --
    ``numpy_adjoint.array`` returns a bare ``ndarray`` -- so values without
    `_ad_iadd` fall back to ``+=``, which stays in place for types that
    implement it and rebinds for those that do not.
    """
    iadd = getattr(current, "_ad_iadd", None)
    if iadd is None:
        return operator.iadd(current, val)
    accumulated = iadd(val)
    return current if accumulated is None else accumulated


class BlockVariable(object):
    """References a block output variable."""

    def __init__(self, output):
        self.output = output
        self.adj_value = None
        self.tlm_value = None
        self.hessian_value = None
        self._checkpoint = None
        self.is_control = False
        self.floating_type = False
        # Helper flags for use during tape traversals.
        self.is_control_dependent = False
        self.is_functional_dependency = False
        # By default assume the variable is created externally to the tape.
        self.creation_timestep = -1
        # The timestep during which this variable was last used as an input.
        self.last_use = -1

    def add_adj_output(self, val):
        if self.adj_value is None:
            self.adj_value = val
        else:
            self.adj_value = _accumulate(self.adj_value, val)

    def add_tlm_output(self, val):
        if self.tlm_value is None:
            self.tlm_value = val
        else:
            self.tlm_value = _accumulate(self.tlm_value, val)

    def add_hessian_output(self, val):
        if self.hessian_value is None:
            self.hessian_value = val
        else:
            self.hessian_value = _accumulate(self.hessian_value, val)

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
        overwrite = self.output._ad_will_add_as_dependency()
        overwrite = bool(overwrite)
        tape = get_working_tape()
        if self.last_use < tape.latest_checkpoint:
            self.save_output(overwrite=overwrite)
        tape.add_to_checkpointable_state(self, self.last_use)
        self.last_use = tape.latest_timestep

    def will_add_as_output(self):
        tape = get_working_tape()
        self.creation_timestep = tape.latest_timestep
        self.last_use = self.creation_timestep
        overwrite = self.output._ad_will_add_as_output()
        overwrite = bool(overwrite)
        if not overwrite:
            self._checkpoint = None
        if tape._eagerly_checkpoint_outputs:
            self.save_output()
        tape.add_to_adjoint_dependencies(self, self.last_use - 1)

    def __str__(self):
        return str(self.output._ad_str)

    @property
    def checkpoint(self):
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value):
        if self.is_control:
            return
        self._checkpoint = value
