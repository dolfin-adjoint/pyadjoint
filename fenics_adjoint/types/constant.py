import backend
from pyadjoint.overloaded_type import OverloadedType


class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)

    def get_derivative(self, options={}):
        return Constant(self.get_adj_output())

    def adj_update_value(self, value):
        self.assign(value)
        self.original_block_output.save_output()

    def _ad_create_checkpoint(self):
        return Constant(self)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return Constant(self*other)

    def _ad_add(self, other):
        return Constant(self+other)

    def _ad_dot(self, other):
        return sum(self.values()*other.values())