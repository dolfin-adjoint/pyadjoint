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

    def create_bc(bc, value=None, homogenize=None):
        """Create a new bc object from an existing one.

        :arg bc: The :class:`~.DirichletBC` to clone.
        :arg value: A new value to use.
        :arg homogenize: If True, return a homogeneous version of the bc.

        One cannot provide both ``value`` and ``homogenize``, but
        should provide at least one.
        """
        if value is None and homogenize is None:
            raise ValueError("No point cloning a bc if you're not changing its values")
        if value is not None and homogenize is not None:
            raise ValueError("Cannot provide both value and homogenize")
        if homogenize:
            value = 0
        return backend.DirichletBC(bc.function_space(),
                                   value,
                                   bc.sub_domain, method=bc.method)

    def function_from_vector(V, vector):
        """Create a new Function sharing data.

        :arg V: The function space
        :arg vector: The data to share.
        """
        return backend.Function(V, val=vector)
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

    def create_bc(bc, value=None, homogenize=None):
        """Create a new bc object from an existing one.

        :arg bc: The :class:`~.DirichletBC` to clone.
        :arg value: A new value to use.
        :arg homogenize: If True, return a homogeneous version of the bc.

        One cannot provide both ``value`` and ``homogenize``, but
        should provide at least one.
        """
        if value is None and homogenize is None:
            raise ValueError("No point cloning a bc if you're not changing its values")
        if value is not None and homogenize is not None:
            raise ValueError("Cannot provide both value and homogenize")
        if homogenize:
            bc = backend.DirichletBC(bc)
            bc.homogenize()
            return bc
        return backend.DirichletBC(bc.function_space(),
                                   value,
                                   *bc.domain_args, method=bc.method())

    def function_from_vector(V, vector):
        """Create a new Function sharing data.

        :arg V: The function space
        :arg vector: The data to share.
        """
        return backend.Function(V, vector)
