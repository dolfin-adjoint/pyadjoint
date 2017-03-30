import backend
from pyadjoint.overloaded_type import OverloadedType


class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)

    def adj_update_value(self, value):
        if isinstance(value, backend.Constant):
            self.assign(value)
        else:
            # Assume float/integer
            self.assign(Constant(value))