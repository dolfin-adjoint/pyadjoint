from contextlib import contextmanager
from functools import cached_property
import itertools
from numbers import Complex
import warnings
import weakref

import numpy as np

from ..enlisting import Enlist
from ..overloaded_type import OverloadedType
from .optimization_problem import MinimizationProblem
from .optimization_solver import OptimizationSolver


try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None

__all__ = \
    [
        "TAOSolver"
    ]


def attach_destroy_finalizer(obj, *args):
    """Attach a finalizer to `obj` which calls the `destroy` method on each
    element of `args`. Used to avoid potential memory leaks when PETSc objects
    reference themselves via Python callbacks.

    Note: May lead to deadlocks if `obj` is destroyed asynchronously on
    different processes, e.g. due to garbage collection.

    Args:
        X (object): A finalizer is attached to this object.
        args (Sequence[object]): The `destroy` method of each element is
            called when `obj` is destroyed (except at exit). Any `None`
            elements are ignored.
    """

    def finalize_callback(*args):
        for arg in args:
            if arg is not None:
                arg.destroy()

    finalize = weakref.finalize(obj, finalize_callback,
                                *args)
    finalize.atexit = False


class PETScVecInterface:
    """Interface for conversion between :class:`OverloadedType` objects and
    :class:`petsc4py.PETSc.Vec` objects.

    This uses the generic interface provided by :class:`OverloadedType`.

    Args:
        X (OverloadedType or Sequence[OverloadedType]): Defines the data
            layout.
        comm (petsc4py.PETSc.Comm or mpi4py.MPI.Comm): Communicator.
    """

    def __init__(self, X, *, comm=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        X = Enlist(X)
        if comm is None:
            comm = PETSc.COMM_WORLD
        if hasattr(comm, "tompi4py"):
            comm = comm.tompi4py()

        vecs = tuple(x._ad_to_petsc() for x in X)
        n = sum(vec.getLocalSize() for vec in vecs)
        N = sum(vec.getSize() for vec in vecs)
        _, isets = PETSc.Vec().concatenate(vecs)
        for vec in vecs:
            vec.destroy()
        del vecs

        self._comm = comm
        self._n = n
        self._N = N
        self._isets = isets

        attach_destroy_finalizer(self, *self._isets)

    @property
    def comm(self):
        """Communicator.
        """

        return self._comm

    @property
    def n(self):
        """Total number of process local DoFs."""

        return self._n

    @property
    def N(self):
        """Total number of global DoFs."""

        return self._N

    def new_petsc(self):
        """Construct a new :class:`petsc4py.PETSc.Vec`.

        Returns:
            petsc4py.PETSc.Vec: The new :class:`petsc4py.PETSc.Vec`.
        """

        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes((self.n, self.N))
        vec.setUp()
        return vec

    def from_petsc(self, y, X):
        """Copy data from a :class:`petsc4py.PETSc.Vec` to variables.

        Args:
            y (petsc4py.PETSc.Vec): The input :class:`petsc4py.PETSc.Vec`.
            X (OverloadedType or Sequence[OverloadedType]): The output
                variables.
        """

        X = Enlist(X)
        if len(X) != len(self._isets):
            raise ValueError("Invalid length")
        for iset, x in zip(self._isets, X):
            y_sub = y.getSubVector(iset)
            x._ad_from_petsc(y_sub)
            y.restoreSubVector(iset, y_sub)

    def to_petsc(self, x, Y):
        """Copy data to a :class:`petsc4py.PETSc.Vec`.

        Args:
            x (petsc4py.PETSc.Vec): The output :class:`petsc4py.PETSc.Vec`.
            Y (Complex, OverloadedType or Sequence[OverloadedType]):
                Values for input variables.
        """

        Y = Enlist(Y)
        if len(Y) != len(self._isets):
            raise ValueError("Invalid length")
        for iset, y in zip(self._isets, Y):
            x_sub = x.getSubVector(iset)
            if isinstance(y, Complex):
                x_sub.set(y)
            elif isinstance(y, OverloadedType):
                y._ad_to_petsc(vec=x_sub)
            else:
                raise TypeError(f"Unexpected type: {type(y)}")
            x_sub.restoreSubVector(iset, x_sub)


# Modified version of flatten_parameters function from firedrake/petsc.py,
# Firedrake master branch 57e21cc8ebdb044c1d8423b48f3dbf70975d5548, first
# added 2024-08-08
def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python3

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python3

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in parameters.items():
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            warnings.warn(("Ignoring duplicate option: %s (existing value %s, new value %s)")
                          % (option, new[option], value))
        new[option] = value
    return new


# Modified version of OptionsManager class from firedrake/petsc.py,
# Firedrake master branch 57e21cc8ebdb044c1d8423b48f3dbf70975d5548, first
# added 2024-08-08
class OptionsManager(object):

    # What appeared on the commandline, we should never clear these.
    # They will override options passed in as a dict if an
    # options_prefix was supplied.
    commandline_options = frozenset(PETSc.Options().getAll())

    options_object = PETSc.Options()

    count = itertools.count()

    """Mixin class that helps with managing setting petsc options.

    :arg parameters: The dictionary of parameters to use.
    :arg options_prefix: The prefix to look up items in the global
        options database (may be ``None``, in which case only entries
        from ``parameters`` will be considered.  If no trailing
        underscore is provided, one is appended.  Hence ``foo_`` and
        ``foo`` are treated equivalently.  As an exception, if the
        prefix is the empty string, no underscore is appended.

    To use this, you must call its constructor to with the parameters
    you want in the options database.

    You then call :meth:`set_from_options`, passing the PETSc object
    you'd like to call ``setFromOptions`` on.  Note that this will
    actually only call ``setFromOptions`` the first time (so really
    this parameters object is a once-per-PETSc-object thing).

    So that the runtime monitors which look in the options database
    actually see options, you need to ensure that the options database
    is populated at the time of a ``SNESSolve`` or ``KSPSolve`` call.
    Do that using the :meth:`inserted_options` context manager.

    .. code-block:: python3

       with self.inserted_options():
           self.snes.solve(...)

    This ensures that the options database has the relevant entries
    for the duration of the ``with`` block, before removing them
    afterwards.  This is a much more robust way of dealing with the
    fixed-size options database than trying to clear it out using
    destructors.

    This object can also be used only to manage insertion and deletion
    into the PETSc options database, by using the context manager.
    """
    def __init__(self, parameters, options_prefix):
        super().__init__()
        if parameters is None:
            parameters = {}
        else:
            # Convert nested dicts
            parameters = flatten_parameters(parameters)
        if options_prefix is None:
            self.options_prefix = "pyadjoint_%d_" % next(self.count)
            self.parameters = parameters
            self.to_delete = set(parameters)
        else:
            if len(options_prefix) and not options_prefix.endswith("_"):
                options_prefix += "_"
            self.options_prefix = options_prefix
            # Remove those options from the dict that were passed on
            # the commandline.
            self.parameters = {k: v for k, v in parameters.items()
                               if options_prefix + k not in self.commandline_options}
            self.to_delete = set(self.parameters)
            # Now update parameters from options, so that they're
            # available to solver setup (for, e.g., matrix-free).
            # Can't ask for the prefixed guy in the options object,
            # since that does not DTRT for flag options.
            for k, v in self.options_object.getAll().items():
                if k.startswith(self.options_prefix):
                    self.parameters[k[len(self.options_prefix):]] = v
        self._setfromoptions = False

    def set_default_parameter(self, key, val):
        """Set a default parameter value.

        :arg key: The parameter name
        :arg val: The parameter value.

        Ensures that the right thing happens cleaning up the options
        database.
        """
        k = self.options_prefix + key
        if k not in self.options_object and key not in self.parameters:
            self.parameters[key] = val
            self.to_delete.add(key)

    def set_from_options(self, petsc_obj):
        """Set up petsc_obj from the options database.

        :arg petsc_obj: The PETSc object to call setFromOptions on.

        Matt says: "Only ever call setFromOptions once".  This
        function ensures we do so.
        """
        if not self._setfromoptions:
            with self.inserted_options():
                petsc_obj.setOptionsPrefix(self.options_prefix)
                # Call setfromoptions inserting appropriate options into
                # the options database.
                petsc_obj.setFromOptions()
                self._setfromoptions = True

    @contextmanager
    def inserted_options(self):
        """Context manager inside which the petsc options database
    contains the parameters from this object."""
        try:
            for k, v in self.parameters.items():
                self.options_object[self.options_prefix + k] = v
            yield
        finally:
            for k in self.to_delete:
                del self.options_object[self.options_prefix + k]


class TAOObjective:
    """Utility class for computing functional values and associated
    derivatives.

    Args:
        rf (ReducedFunctional): Defines the forward, and used to compute
            derivative information.
    """

    def __init__(self, rf):
        self._reduced_functional = rf

    @property
    def reduced_functional(self):
        """:class:`.ReducedFunctional`. Defines the forward, and used to
        compute derivative information.
        """

        return self._reduced_functional

    def objective_gradient(self, M):
        """Evaluate the forward, and compute a first derivative.

        Args:
            M (OverloadedType or Sequence[OverloadedType]): Defines the control
                value.
        Returns:
            AdjFloat: The value of the functional.
            OverloadedType or Sequence[OverloadedType]: The (dual space)
                derivative.
        """

        M = Enlist(M)
        J = self.reduced_functional(tuple(m._ad_copy() for m in M))
        _ = self.reduced_functional.derivative()
        dJ = tuple(m.control.block_variable.adj_value
                   for m in self.reduced_functional.controls)
        return J, M.delist(dJ)

    def hessian(self, M, M_dot):
        """Evaluate the forward, and compute a second derivative action on a
        given direction.

        Args:
            M (OverloadedType or Sequence[OverloadedType]): Defines the control
                value.
            M_dot (OverloadedType or Sequence[OverloadedType]): Defines the
                direction.
        Returns:
            OverloadedType or Sequence[OverloadedType]: The (dual space)
                second derivative action on the given direction.
        """

        M = Enlist(M)
        M_dot = Enlist(M_dot)
        _ = self.reduced_functional(tuple(m._ad_copy() for m in M))
        _ = self.reduced_functional.derivative()
        _ = self.reduced_functional.hessian(tuple(m_dot._ad_copy() for m_dot in M_dot))
        ddJ = tuple(m.control.block_variable.hessian_value
                    for m in self.reduced_functional.controls)
        return M.delist(ddJ)

    def new_M(self):
        """Return a new variable or variables suitable for storing a control
        value. Not initialized to zero.

        Returns:
            OverloadedType or Sequence[OverloadedType]: New variable or
                variables suitable for storing a control value.
        """

        # Not initialized to zero
        return tuple(m.control._ad_copy()
                     for m in self.reduced_functional.controls)

    def new_M_dual(self):
        """Return a new variable or variables suitable for storing a value for
        a (dual space) derivative of the functional with respect to the
        control. Not initialized to zero.

        Returns:
            OverloadedType or Sequence[OverloadedType]: New variable or
                variables suitable for storing a value for a (dual space)
                derivative of the functional with respect to the control.
        """

        # Not initialized to zero, requires adjoint or Hessian action values to
        # have already been computed
        M_dual = []
        for m in self.reduced_functional.controls:
            bv = m.control.block_variable
            if bv.adj_value is not None:
                M_dual.append(bv.adj_value)
            elif bv.hessian_value is not None:
                M_dual.append(bv.hessian_value)
            else:
                raise RuntimeError("Unable to instantiate new dual space "
                                   "object")
        return tuple(m_dual._ad_copy() for m_dual in M_dual)


class TAOSolver(OptimizationSolver):
    """Use TAO to solve an optimization problem.

    Args:
        problem (MinimizationProblem): Defines the optimization problem to be
            solved.
        parameters (Mapping): TAO options.
        comm (petsc4py.PETSc.Comm or mpi4py.MPI.Comm): Communicator.
        convert_options (Mapping): Defines the `options` argument to
            :meth:`OverloadedType._ad_convert_type`.
    """

    def __init__(self, problem, parameters, *, comm=None, convert_options=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        if not isinstance(problem, MinimizationProblem):
            raise TypeError("MinimizationProblem required")
        if problem.constraints is not None:
            raise NotImplementedError("Constraints not implemented")

        if comm is None:
            comm = PETSc.COMM_WORLD
        if hasattr(comm, "tompi4py"):
            comm = comm.tompi4py()
        if convert_options is None:
            convert_options = {}

        taoobjective = TAOObjective(problem.reduced_functional)

        vec_interface = PETScVecInterface(
            tuple(m.control for m in taoobjective.reduced_functional.controls),
            comm=comm)
        n, N = vec_interface.n, vec_interface.N
        to_petsc, from_petsc = vec_interface.to_petsc, vec_interface.from_petsc
        tao = PETSc.TAO().create(comm=comm)

        def objective_gradient(tao, x, g):
            M = taoobjective.new_M()
            from_petsc(x, M)
            J_val, dJ = taoobjective.objective_gradient(M)
            to_petsc(g, dJ)
            return J_val

        tao.setObjectiveGradient(objective_gradient, None)

        def hessian(tao, x, H, P):
            H.getPythonContext().set_M(x)

        class Hessian:
            def __init__(self):
                self._shift = 0.0

            @cached_property
            def _M(self):
                return taoobjective.new_M()

            def set_M(self, x):
                from_petsc(x, self._M)
                self._shift = 0.0

            def shift(self, A, alpha):
                self._shift += alpha

            def mult(self, A, x, y):
                M_dot = taoobjective.new_M()
                from_petsc(x, M_dot)
                ddJ = taoobjective.hessian(self._M, M_dot)
                to_petsc(y, ddJ)
                if self._shift != 0.0:
                    y.axpy(self._shift, x)

        H_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                            Hessian(), comm=comm)
        H_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        H_matrix.setUp()
        tao.setHessian(hessian, H_matrix)

        class GradientNorm:
            def mult(self, A, x, y):
                X = taoobjective.new_M_dual()
                from_petsc(x, X)
                assert len(taoobjective.reduced_functional.controls) == len(X)
                X = tuple(m._ad_convert_type(x, options=convert_options)
                          for m, x in zip(taoobjective.reduced_functional.controls, X))
                to_petsc(y, X)

        M_inv_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                GradientNorm(), comm=comm)
        M_inv_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        M_inv_matrix.setUp()
        tao.setGradientNorm(M_inv_matrix)

        if problem.bounds is not None:
            lbs = []
            ubs = []
            assert len(problem.bounds) == len(problem.reduced_functional.controls)
            for (lb, ub), control in zip(problem.bounds, taoobjective.reduced_functional.controls):
                if lb is None:
                    lb = np.finfo(PETSc.ScalarType).min
                if ub is None:
                    ub = np.finfo(PETSc.ScalarType).max
                lbs.append(lb)
                ubs.append(ub)

            lb_vec = vec_interface.new_petsc()
            ub_vec = vec_interface.new_petsc()
            to_petsc(lb_vec, lbs)
            to_petsc(ub_vec, ubs)
            tao.setVariableBounds(lb_vec, ub_vec)
            lb_vec.destroy()
            ub_vec.destroy()

        options = OptionsManager(parameters, None)
        options.set_from_options(tao)

        if tao.getType() in {PETSc.TAO.Type.LMVM, PETSc.TAO.Type.BLMVM}:
            class InitialHessian:
                pass

            class InitialHessianPreconditioner:
                def apply(self, pc, x, y):
                    X = taoobjective.new_M_dual()
                    from_petsc(x, X)
                    assert len(taoobjective.reduced_functional.controls) == len(X)
                    X = tuple(m._ad_convert_type(x, options=convert_options)
                              for m, x in zip(taoobjective.reduced_functional.controls, X))
                    to_petsc(y, X)

            B_0_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                  InitialHessian(), comm=comm)
            B_0_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            B_0_matrix.setUp()

            B_0_matrix_pc = PETSc.PC().createPython(InitialHessianPreconditioner(),
                                                    comm=comm)
            B_0_matrix_pc.setOperators(B_0_matrix)
            B_0_matrix_pc.setUp()

            tao.setLMVMH0(B_0_matrix)
            ksp = tao.getLMVMH0KSP()
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.setTolerances(rtol=0.0, atol=0.0, divtol=None, max_it=1)
            ksp.setPC(B_0_matrix_pc)
            ksp.setUp()
        else:
            B_0_matrix = None
            B_0_matrix_pc = None

        x = vec_interface.new_petsc()
        tao.setSolution(x)
        tao.setUp()

        super().__init__(problem, parameters)
        self._taoobjective = taoobjective
        self._vec_interface = vec_interface
        self._tao = tao
        self._x = x

        attach_destroy_finalizer(
            self, tao, H_matrix, M_inv_matrix, B_0_matrix_pc, B_0_matrix, x)

    @property
    def taoobjective(self):
        """The :class:`.TAOObjective` used for the optimization.
        """

        return self._taoobjective

    @property
    def tao(self):
        """The :class:`petsc4py.PETSc.TAO` used for the optimization.
        """

        return self._tao

    @property
    def x(self):
        """The :class:`petsc4py.PETSc.Vec` used to store the solution to the
        optimization problem.
        """

        return self._x

    def solve(self):
        """Solve the optimization problem.

        Returns:
            OverloadedType or tuple[OverloadedType]: The solution.
        """
        M = tuple(
            m.tape_value()._ad_copy()
            for m in self.taoobjective.reduced_functional.controls)
        self._vec_interface.to_petsc(self.x, M)
        self.tao.solve()
        self._vec_interface.from_petsc(self.x, M)
        if self.tao.getConvergedReason() <= 0:
            raise RuntimeError("Convergence failure")
        return self.taoobjective.reduced_functional.controls.delist(M)
