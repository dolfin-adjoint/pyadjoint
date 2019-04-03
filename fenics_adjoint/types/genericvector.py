import backend
from .compat import gather

__all__ = []


@staticmethod
def _ad_to_list(self):
    return gather(self)


backend.GenericVector._ad_to_list = _ad_to_list
