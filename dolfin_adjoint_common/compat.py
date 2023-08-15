
class Compat:
    # A bag class to act as a namespace for compat.
    pass


def compat(backend):
    compat = Compat()

    if backend.__name__ == "firedrake":
        raise NotImplementedError("backend cannot be firedrake.")
    else:
        compat.Expression = backend.Expression
        compat.MatrixType = (backend.cpp.la.Matrix, backend.cpp.la.GenericMatrix)
        compat.VectorType = backend.cpp.la.GenericVector
        compat.FunctionType = backend.cpp.function.Function
        compat.FunctionSpace = backend.FunctionSpace
        compat.FunctionSpaceType = backend.cpp.function.FunctionSpace
        compat.ExpressionType = backend.function.expression.BaseExpression

        compat.MeshType = backend.Mesh

        compat.backend_fs_sub = backend.FunctionSpace.sub

        def _fs_sub(self, i):
            V = compat.backend_fs_sub(self, i)
            V._ad_parent_space = self
            return V
        backend.FunctionSpace.sub = _fs_sub

        compat.backend_fs_collapse = backend.FunctionSpace.collapse

        def _fs_collapse(self, collapsed_dofs=False):
            """Overloaded FunctionSpace.collapse to limit the amount of MPI communicator created.
            """
            if not hasattr(self, "_ad_collapsed_space"):
                # Create collapsed space
                self._ad_collapsed_space = compat.backend_fs_collapse(self, collapsed_dofs=True)

            if collapsed_dofs:
                return self._ad_collapsed_space
            else:
                return self._ad_collapsed_space[0]
        compat.FunctionSpace.collapse = _fs_collapse

        def extract_subfunction(u, V):
            component = V.component()
            r = u
            for idx in component:
                r = r.sub(int(idx))
            return r
        compat.extract_subfunction = extract_subfunction

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
                bc = backend.DirichletBC(backend.FunctionSpace(bc.function_space()),
                                         value, *bc.domain_args)
            except AttributeError:
                bc = backend.DirichletBC(backend.FunctionSpace(bc.function_space()),
                                         value,
                                         bc.sub_domain, method=bc.method())
            return bc
        compat.create_bc = create_bc

        def function_from_vector(V, vector, cls=backend.Function):
            """Create a new Function from a vector.

            :arg V: The function space
            :arg vector: The vector data.
            """
            if isinstance(vector, backend.cpp.la.PETScVector)\
               or isinstance(vector, backend.cpp.la.Vector):
                pass
            elif not isinstance(vector, backend.Vector):
                # If vector is a fenics_adjoint.Function, which does not inherit
                # backend.cpp.function.Function with pybind11
                vector = vector._cpp_object
            r = cls(V)
            r.vector()[:] = vector
            return r
        compat.function_from_vector = function_from_vector

        def inner(a, b):
            """Compute the l2 inner product of a and b.

            :arg a: a Vector.
            :arg b: a Vector.
            """
            return a.inner(b)
        compat.inner = inner

        def extract_bc_subvector(value, Vtarget, bc):
            """Extract from value (a function in a mixed space), the sub
            function corresponding to the part of the space bc applies
            to.  Vtarget is the target (collapsed) space."""
            assigner = backend.FunctionAssigner(Vtarget, backend.FunctionSpace(bc.function_space()))
            output = backend.Function(Vtarget)
            assigner.assign(output, extract_subfunction(value, backend.FunctionSpace(bc.function_space())))
            return output.vector()
        compat.extract_bc_subvector = extract_bc_subvector

        def extract_mesh_from_form(form):
            """Takes in a form and extracts a mesh which can be used to construct function spaces.

            Dolfin only accepts dolfin.cpp.mesh.Mesh types for function spaces, while firedrake use ufl.Mesh.

            Args:
                form (ufl.Form): Form to extract mesh from

            Returns:
                dolfin.Mesh: The extracted mesh

            """
            return form.ufl_domain().ufl_cargo()
        compat.extract_mesh_from_form = extract_mesh_from_form

        def constant_function_firedrake_compat(value):
            """Only needed on firedrake side.

            See docstring for the firedrake version of this function above.
            """
            return value
        compat.constant_function_firedrake_compat = constant_function_firedrake_compat

        def assemble_adjoint_value(*args, **kwargs):
            """Wrapper that assembles a matrix with boundary conditions"""
            bcs = kwargs.pop("bcs", ())
            result = backend.assemble(*args, **kwargs)
            for bc in bcs:
                bc.apply(result)
            return result
        compat.assemble_adjoint_value = assemble_adjoint_value

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
        compat.gather = gather

        def linalg_solve(A, x, b, *args, **kwargs):
            """Linear system solve that has a firedrake compatible interface.

            Throws away kwargs and uses b.vector() as RHS if
            b is not a GenericVector instance.

            """
            if not isinstance(b, backend.GenericVector):
                b = b.vector()
            return backend.solve(A, x, b, *args)
        compat.linalg_solve = linalg_solve

        def type_cast_function(obj, cls):
            """Type casts Function object `obj` to an instance of `cls`.

            Useful when converting backend.Function to overloaded Function.
            """
            return cls(obj.function_space(), obj._cpp_object)
        compat.type_cast_function = type_cast_function

        def create_constant(*args, **kwargs):
            """Initialise a fenics_adjoint.Constant object and return it."""
            from fenics_adjoint import Constant
            # Dolfin constants do not have domains
            _ = kwargs.pop("domain", None)
            return Constant(*args, **kwargs)
        compat.create_constant = create_constant

        def create_function(*args, **kwargs):
            """Initialises a fenics_adjoint.Function object and returns it."""
            from fenics_adjoint import Function
            return Function(*args, **kwargs)
        compat.create_function = create_function

        def isconstant(expr):
            """Check whether expression is constant type."""
            return isinstance(expr, backend.Constant)
        compat.isconstant = isconstant

    return compat
