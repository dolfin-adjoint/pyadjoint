import backend
from dolfin_adjoint_common import compat
compat = compat.compat(backend)


extract_subfunction = compat.extract_subfunction

__all__ = ["extract_subfunction"]
