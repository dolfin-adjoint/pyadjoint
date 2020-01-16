from dolfin_adjoint_common.compat import compat
import fenics


class Backend:
    backend = fenics
    compat = compat(fenics)
