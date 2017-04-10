import backend
from pyadjoint.overloaded_type import OverloadedType


class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)

    def get_derivative(self):
        return Constant(self.get_adj_output())

    def adj_update_value(self, value):
        if isinstance(value, backend.Constant):
            self.assign(value)
        else:
            # Assume float/integer
            self.assign(Constant(value))

    def _ad_mult(self, other):
        return Constant(self*other)

    def _ad_add(self, other):
        return Constant(self+other)

    def _ad_dot(self, other):
        # TODO: Can Constants have values arrays of size bigger than 1?
        return self.values()[0]*other.values()[0]