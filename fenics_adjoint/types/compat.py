from fenics_adjoint import backend


if backend.__name__ == "firedrake":
    MatrixType = backend.matrix.MatrixBase
    VectorType = backend.vector.Vector
    FunctionType = backend.Function
    FunctionSpaceType = (backend.functionspaceimpl.FunctionSpace,
                         backend.functionspaceimpl.WithGeometry,
                         backend.functionspaceimpl.MixedFunctionSpace)

    FunctionSpace = backend.FunctionSpace

    backend.functionspaceimpl.FunctionSpace._ad_parent_space = property(lambda self: self.parent)

    def extract_subfunction(u, V):
        r = u
        while V.index:
            r = r.sub(V.index)
            V = V.parent
        return r

    def new_bc(bc):
        return type(bc)(bc.function_space(), bc.function_arg, bc.sub_domain,
                        method=bc.method)

    def copy_function(function):
        return backend.Function(function)
else:
    MatrixType = (backend.cpp.Matrix, backend.GenericMatrix)
    VectorType = backend.cpp.la.GenericVector
    FunctionType = backend.cpp.Function
    FunctionSpaceType = backend.cpp.FunctionSpace

    class FunctionSpace(backend.FunctionSpace):
        def sub(self, i):
            V = backend.FunctionSpace.sub(self, i)
            V._ad_parent_space = self
            return V

    def extract_subfunction(u, V):
        component = V.component()
        r = u
        for idx in component:
            r = r.sub(int(idx))
        return r

    def new_bc(bc):
        return backend.DirichletBC(bc)

    def copy_function(function):
        return backend.Function(function.function_space(), function)
