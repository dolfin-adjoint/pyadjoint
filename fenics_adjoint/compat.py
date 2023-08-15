from dolfin_adjoint_common.compat import compat
import dolfin


class Backend:
    backend = dolfin
    compat = compat(dolfin)
