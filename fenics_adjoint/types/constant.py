import backend
from pyadjoint.overloaded_type import OverloadedType


class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)

    def get_derivative(self, adj_value, options={}):
        return Constant(adj_value)

    def adj_update_value(self, value):
        self.original_block_output.checkpoint = value._ad_create_checkpoint()

    def _ad_create_checkpoint(self):
        if self.ufl_shape == ():
            return Constant(self)
        return Constant(self.values())

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return Constant(self*other)

    def _ad_add(self, other):
        return Constant(self+other)

    def _ad_dot(self, other):
        return sum(self.values()*other.values())
