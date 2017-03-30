import backend
import ufl
from pyadjoint.overloaded_type import OverloadedType


class DirichletBC(OverloadedType, backend.DirichletBC):
    def __init__(self, *args, **kwargs):
        super(DirichletBC, self).__init__(*args, **kwargs)
        backend.DirichletBC.__init__(self, *args, **kwargs)

    def adj_update_value(self, value):
        if isinstance(value, backend.DirichletBC):
            self.set_value(value.value())
        elif isinstance(value, ufl.algebra.Sum):
            # TODO: If we decide for something like this,
            # then it would make sense to create our own
            # algebra classes to extract the types we want.

            # This looks pretty ugly as it stands.
            # It would look much better with a class to handle all of it,
            # but it might just be counterproductive to take this approach.
            a = value.ufl_operands[0]
            b = value.ufl_operands[1]
            if isinstance(a, backend.Function) and isinstance(b, ufl.algebra.Product):
                c = a.copy(deepcopy=True)
                c.vector()[:] += b.ufl_operands[0].value()*b.ufl_operands[1].vector()
            self.set_value(c)
        else:
            self.set_value(value)
