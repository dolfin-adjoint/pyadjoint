import backend
import ufl

from dolfin_adjoint_common import compat
compat = compat.compat(backend)
from .function import Function

from pyadjoint.tape import no_annotations
from pyadjoint.overloaded_type import OverloadedType, FloatingType
from fenics_adjoint.blocks import DirichletBCBlock


# TODO: Might need/want some way of creating a new DirichletBCBlock if DirichletBC is assigned
#       new boundary values/function.


class DirichletBC(FloatingType, backend.DirichletBC):
    def __init__(self, *args, **kwargs):
        super(DirichletBC, self).__init__(*args, **kwargs)

        FloatingType.__init__(self,
                              *args,
                              block_class=DirichletBCBlock,
                              _ad_args=args,
                              _ad_floating_active=True,
                              annotate=kwargs.pop("annotate", True),
                              **kwargs)

        # Call backend constructor after popped AD specific keyword args.
        backend.DirichletBC.__init__(self, *args, **kwargs)

        self._g = args[1]
        self._ad_args = args
        self._ad_kwargs = kwargs

    @no_annotations
    def apply(self, *args, **kwargs):
        for arg in args:
            if not hasattr(arg, "bcs"):
                arg.bcs = []
            arg.bcs.append(self)
        return backend.DirichletBC.apply(self, *args, **kwargs)

    def _ad_create_checkpoint(self):
        deps = self.block.get_dependencies()
        if len(deps) <= 0:
            # We don't have any dependencies so the supplied value was not an OverloadedType.
            # Most probably it was just a float that is immutable so will never change.
            return None

        return deps[0]

    def _ad_restore_at_checkpoint(self, checkpoint):
        if checkpoint is not None:
            self.set_value(checkpoint.saved_output)
        return self
