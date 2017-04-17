import backend
from pyadjoint.overloaded_type import OverloadedType


class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)

    def get_derivative(self, project=False):
        return Constant(self.get_adj_output())

    # TODO: Depending on re-computation approach, but as it stands:
    #       - Make adj_update_value update saved output.
    #       - Make checkpoints actually save a weak/deep copy.
    #         (Probably weak unless someone knows a good deepcopy method for constants)
    # If you fix one of these you must also fix the other, otherwise the ReducedFunctional won't work for constants.
    def adj_update_value(self, value):
        if isinstance(value, backend.Constant):
            self.assign(value)
        else:
            # Assume float/integer
            self.assign(Constant(value))

    def _ad_create_checkpoint(self):
        return self

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return Constant(self*other)

    def _ad_add(self, other):
        return Constant(self+other)

    def _ad_dot(self, other):
        # TODO: Can Constants have values arrays of size bigger than 1?
        return self.values()[0]*other.values()[0]