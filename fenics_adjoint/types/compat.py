from fenics_adjoint import backend


if backend.__name__ == "firedrake":
    MatrixType = backend.matrix.MatrixBase
    VectorType = backend.vector.Vector
    FunctionType = backend.Function
    FunctionSpaceType = (backend.functionspaceimpl.FunctionSpace,
                         backend.functionspaceimpl.WithGeometry,
                         backend.functionspaceimpl.MixedFunctionSpace)
    ExpressionType = backend.Expression

    FunctionSpace = backend.FunctionSpace

    backend.functionspaceimpl.FunctionSpace._ad_parent_space = property(lambda self: self.parent)

    backend.functionspaceimpl.WithGeometry._ad_parent_space = property(lambda self: self.parent)


    def extract_subfunction(u, V):
        """If V is a subspace of the function-space of u, return the component of u that is in that subspace."""
        if V.index is not None:
            # V is an indexed subspace of a MixedFunctionSpace
            return u.sub(V.index)
        elif V.component is not None:
            # V is a vector component subspace.
            # The vector functionspace V.parent may itself be a subspace
            # so call this function recursively
            return extract_subfunction(u, V.parent).sub(V.component)
        else:
            return u

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
        return bc.reconstruct(g=value)

    # Most of this is to deal with Firedrake assembly returning
    # Function whereas Dolfin returns Vector.
    def function_from_vector(V, vector):
        """Create a new Function sharing data.

        :arg V: The function space
        :arg vector: The data to share.
        """
        return backend.Function(V, val=vector)

    def inner(a, b):
        """Compute the l2 inner product of a and b.

        :arg a: a Function.
        :arg b: a Vector.
        """
        return a.vector().inner(b)

    def extract_bc_subvector(value, Vtarget, bc):
        """Extract from value (a function in a mixed space), the sub
        function corresponding to the part of the space bc applies
        to.  Vtarget is the target (collapsed) space."""
        r = value
        for idx in bc._indices:
            r = r.sub(idx)
        assert Vtarget == r.function_space()
        return r.vector()

    def extract_mesh_from_form(form):
        """Takes in a form and extracts a mesh which can be used to construct function spaces.

        Dolfin only accepts dolfin.cpp.mesh.Mesh types for function spaces, while firedrake use ufl.Mesh.

        Args:
            form (ufl.Form): Form to extract mesh from

        Returns:
            ufl.Mesh: The extracted mesh

        """
        return form.ufl_domain()

    def constant_function_firedrake_compat(value):
        """Takes a Function/vector and returns the array.

        The Function should belong to the space of Reals.
        This function is needed because Firedrake does not
        accept a Function as argument to Constant constructor.
        It does accept vector (which is what we work with in dolfin),
        but since we work with Functions instead of vectors in firedrake,
        this function call is needed in firedrake_adjoint.

        Args:
            value (Function): A Function to convert

        Returns:
            numpy.ndarray: A numpy array of the function values.

        """
        return value.dat.data

    def assemble_adjoint_value(*args, **kwargs):
        """A wrapper around Firedrake's assemble that returns a Vector 
        instead of a Function when assembling a 1-form."""
        result = backend.assemble(*args, **kwargs)
        if isinstance(result, backend.Function):
            return result.vector()
        else:
            return result

    def gather(vec):
        return vec.gather()

    linalg_solve = backend.solve

else:
    MatrixType = (backend.cpp.la.Matrix, backend.cpp.la.GenericMatrix)
    VectorType = backend.cpp.la.GenericVector
    FunctionType = backend.cpp.function.Function
    FunctionSpaceType = backend.cpp.function.FunctionSpace
    ExpressionType = backend.function.expression.BaseExpression

    backend_fs_sub = backend.FunctionSpace.sub
    def _fs_sub(self, i):
        V = backend_fs_sub(self, i)
        V._ad_parent_space = self
        return V
    backend.FunctionSpace.sub = _fs_sub

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
        try:
            # FIXME: Not perfect handling of Initialization, wait for development in dolfin.DirihcletBC
            bc =  backend.DirichletBC(backend.FunctionSpace(bc.function_space()),
                                      value, *bc.domain_args)
        except AttributeError:
            bc = backend.DirichletBC(backend.FunctionSpace(bc.function_space()),
                                     value,
                                     bc.sub_domain, method=bc.method())
        return bc

    def function_from_vector(V, vector):
        """Create a new Function from a vector.

        :arg V: The function space
        :arg vector: The vector data.
        """
        if isinstance(vector, backend.cpp.la.PETScVector)\
           or  isinstance(vector, backend.cpp.la.Vector):
            pass
        elif not isinstance(vector, backend.Vector):
            # If vector is a fenics_adjoint.Function, which does not inherit
            # backend.cpp.function.Function with pybind11
            vector = vector._cpp_object
        r = backend.Function(V)
        r.vector()[:] = vector
        return r

    def inner(a, b):
        """Compute the l2 inner product of a and b.

        :arg a: a Vector.
        :arg b: a Vector.
        """
        return a.inner(b)

    def extract_bc_subvector(value, Vtarget, bc):
        """Extract from value (a function in a mixed space), the sub
        function corresponding to the part of the space bc applies
        to.  Vtarget is the target (collapsed) space."""
        assigner = backend.FunctionAssigner(Vtarget, backend.FunctionSpace(bc.function_space()))
        output = backend.Function(Vtarget)
        assigner.assign(output, extract_subfunction(value, backend.FunctionSpace(bc.function_space())))
        return output.vector()

    def extract_mesh_from_form(form):
        """Takes in a form and extracts a mesh which can be used to construct function spaces.

        Dolfin only accepts dolfin.cpp.mesh.Mesh types for function spaces, while firedrake use ufl.Mesh.

        Args:
            form (ufl.Form): Form to extract mesh from

        Returns:
            dolfin.Mesh: The extracted mesh

        """
        return form.ufl_domain().ufl_cargo()

    def constant_function_firedrake_compat(value):
        """Only needed on firedrake side.

        See docstring for the firedrake version of this function above.
        """
        return value

    assemble_adjoint_value = backend.assemble

    def gather(vec):
        import numpy
        if isinstance(vec, backend.cpp.function.Function):
            vec = vec.vector()

        if isinstance(vec, backend.cpp.la.GenericVector):
            arr = vec.gather(numpy.arange(vec.size(), dtype='I'))
        elif isinstance(vec, list):
            return list(map(gather, vec))
        else:
            arr = vec  # Assume it's a gathered numpy array already

        return arr

    def linalg_solve(*args, **kwargs):
        """Temporary workaround for kwargs not expected in fenics linalg,
        but possible in firedrake.

        A better solution is expected in the future.

        """
        return backend.solve(*args)
