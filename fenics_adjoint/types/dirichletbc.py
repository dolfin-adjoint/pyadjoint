import backend
from pyadjoint.tape import OverloadedType


class DirichletBC(OverloadedType, backend.DirichletBC):
    def __init__(self, *args, **kwargs):
        super(DirichletBC, self).__init__(*args, **kwargs)
        backend.DirichletBC.__init__(self, *args, **kwargs)
