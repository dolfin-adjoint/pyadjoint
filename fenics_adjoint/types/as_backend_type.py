import backend

_backend_as_backend_type = backend.as_backend_type


def as_backend_type(A):
    out = _backend_as_backend_type(A)
    out._ad_original_ref = A
    return out


__set_nullspace = backend.cpp.la.PETScMatrix.set_nullspace


def set_nullspace(self, nullspace):
    self._ad_original_ref._ad_nullspace = nullspace
    __set_nullspace(self, nullspace)


backend.cpp.la.PETScMatrix.set_nullspace = set_nullspace


class VectorSpaceBasis(backend.VectorSpaceBasis):
    def __init__(self, *args, **kwargs):
        super(VectorSpaceBasis, self).__init__(*args, **kwargs)
        self._ad_orthogonalized = False

    def orthogonalize(self, vector):
        backend.VectorSpaceBasis.orthogonalize(self, vector)
        self._ad_orthogonalized = True
