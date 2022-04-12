from .tape import no_annotations
from functools import wraps
_stop_saving = 0


def pause_saving():
    global _stop_saving
    _stop_saving += 1


def continue_saving():
    global _stop_saving
    _stop_saving -= 1
    return _stop_saving <= 0


class stop_saving(object):
    def __enter__(self):
        pause_saving()

    def __exit__(self, *args):
        continue_saving()


def no_saving(function):
    """Decorator to turn off saving outputs for the decorated function."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with stop_saving():
            return function(*args, **kwargs)

    return wrapper


def save_outputs(kwargs=None):
    """Returns True if saving flag is on, and False if not.

    If kwargs is given, the function will try to extract the
    save_outputs keyword. If the save_outputs keyword is not present it defaults to True.
    If saving has been paused, then it will always return False.

    Args:
        kwargs (dict): A dictionary of keyword arguments to extract from.
            Note that this should be passed as a dictionary and not actual keyword arguments.

    Returns: bool

    """
    saving = kwargs is None or kwargs.pop("save_outputs", True)

    # TODO: Consider if there is any scenario where one would want the keyword to have
    # precedence over the global flag.
    if _stop_saving > 0:
        return False

    return saving


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
        if save_outputs() and (overwrite or self.checkpoint is None):
            self._checkpoint = self.output._ad_create_checkpoint()

    @property
    def saved_output(self):
        if self.checkpoint is not None:
            return self.output._ad_restore_at_checkpoint(self.checkpoint)
        else:
            return self.output

    def will_add_as_dependency(self):
        overwrite = self.output._ad_will_add_as_dependency()
        overwrite = False if overwrite is None else overwrite
        self.save_output(overwrite=overwrite)

    def will_add_as_output(self):
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
