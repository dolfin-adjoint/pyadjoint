import backend

backend_genericmatrix_mul = backend.cpp.la.GenericMatrix.__mul__


def adjoint_genericmatrix_mul(self, other):
    out = backend_genericmatrix_mul(self, other)
    if hasattr(self, 'form') and isinstance(other, backend.cpp.la.GenericVector):
        if hasattr(other, 'form'):
            out.form = backend.action(self.form, other.form)
        elif hasattr(other, 'function'):
            if hasattr(other, 'function_factor'):
                out.form = backend.action(other.function_factor*self.form, other.function)
            else:
                out.form = backend.action(self.form, other.function)

    return out

backend.cpp.la.GenericMatrix.__mul__ = adjoint_genericmatrix_mul
