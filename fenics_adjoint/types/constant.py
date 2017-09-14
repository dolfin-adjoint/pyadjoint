import backend
from pyadjoint.overloaded_type import OverloadedType
from .compat import constant_function_firedrake_compat


class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)

    def get_derivative(self, options={}):
        return self._ad_convert_type(self.get_adj_output(), options=options)

    def adj_update_value(self, value):
        self.original_block_output.checkpoint = value._ad_create_checkpoint()

    def _ad_convert_type(self, value, options={}):
        value = constant_function_firedrake_compat(value)
        return Constant(value)

    def _ad_function_space(self, mesh):
        element = self.ufl_element()
        args = [element.family(), mesh.ufl_cell(), element.degree()]

        # TODO: In newer versions of FEniCS we may use FiniteElement.reconstruct and avoid the if-test
        #       and just write element.reconstruct(cell=mesh.ufl_cell()).
        if isinstance(element, backend.VectorElement):
            fs_element = element.__class__(*args, dim=len(element.sub_elements()))
        else:
            fs_element = element.__class__(*args)
        return backend.FunctionSpace(mesh, fs_element)

    def _ad_create_checkpoint(self):
        if self.ufl_shape == ():
            return Constant(self)
        return Constant(self.values())

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        values = ufl_shape_workaround(self.values() * other)
        return Constant(values)

    def _ad_add(self, other):
        values = ufl_shape_workaround(self.values() + other.values())
        return Constant(values)

    def _ad_dot(self, other):
        return sum(self.values()*other.values())


def ufl_shape_workaround(values):
    """Workaround because of the following behaviour in FEniCS/Firedrake

    c = Constant(1.0)
    c2 = Constant(c2.values())
    c.ufl_shape == ()
    c2.ufl_shape == (1,)

    Thus you will get a shapes don't match error if you try to replace c with c2 in a UFL form.
    Because of this we require that scalar constants in the forward model are all defined with ufl_shape == (),
    otherwise you will most likely see an error.

    Args:
        values: Array of floats that should come from a Constant.values() call.

    Returns:
        A float if the Constant was scalar, otherwise the original array.

    """
    if len(values) == 1:
        return values[0]
    return values
