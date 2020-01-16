import backend
from dolfin_adjoint_common import compat

from pyadjoint.tape import stop_annotating

compat = compat.compat(backend)

_backend_SystemAssembler_assemble = backend.SystemAssembler.assemble
_backend_SystemAssembler_init = backend.SystemAssembler.__init__


def SystemAssembler_init(self, *args, **kwargs):
    _backend_SystemAssembler_init(self, *args, **kwargs)

    self._A_form = args[0]
    self._b_form = args[1]


def SystemAssembler_assemble(self, *args, **kwargs):
    with stop_annotating():
        out = _backend_SystemAssembler_assemble(self, *args, **kwargs)

    for arg in args:
        if isinstance(arg, compat.VectorType):
            arg.form = self._b_form
            arg.bcs = self._bcs
        elif isinstance(arg, compat.MatrixType):
            arg.form = self._A_form
            arg.bcs = self._bcs
            arg.assemble_system = True
        else:
            raise RuntimeError("Argument type not supported: ", type(arg))
    return out


backend.SystemAssembler.assemble = SystemAssembler_assemble
backend.SystemAssembler.__init__ = SystemAssembler_init
