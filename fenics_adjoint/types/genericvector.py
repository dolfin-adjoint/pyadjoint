import backend
from dolfin_adjoint_common import compat
compat = compat.compat(backend)

__all__ = []


@staticmethod
def _ad_to_list(self):
    return compat.gather(self)


backend.GenericVector._ad_to_list = _ad_to_list
