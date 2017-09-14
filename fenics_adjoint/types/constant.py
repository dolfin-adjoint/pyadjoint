import backend
from pyadjoint.tape import get_working_tape, no_annotations
from pyadjoint.overloaded_type import OverloadedType
from .compat import constant_function_firedrake_compat
from pyadjoint.block import Block
from .types import create_overloaded_object


class Constant(OverloadedType, backend.Constant):
    def __init__(self, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        backend.Constant.__init__(self, *args, **kwargs)

    def assign(self, *args, **kwargs):
        annotate_tape = kwargs.pop("annotate_tape", True)
        if annotate_tape:
            other = args[0]
            if not isinstance(other, OverloadedType):
                other = create_overloaded_object(other)

            block = AssignBlock(self, other)
            tape = get_working_tape()
            tape.add_block(block)

        ret = backend.Constant.assign(self, *args, **kwargs)

        if annotate_tape:
            block.add_output(self.create_block_output())

        return ret

    def _ad_convert_type(self, value, options={}):
        return Constant(value)

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


class AssignBlock(Block):
    def __init__(self, func, other):
        super(AssignBlock, self).__init__()
        self.add_dependency(func.get_block_output())
        self.add_dependency(other.get_block_output())

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()
        self.get_dependencies()[1].add_adj_output(adj_input)

    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[1].tlm_value
        self.get_outputs()[0].add_tlm_output(tlm_input)

    def evaluate_hessian(self):
        hessian_input = self.get_outputs()[0].hessian_value
        self.get_dependencies()[1].add_hessian_output(hessian_input)

    def recompute(self):
        deps = self.get_dependencies()
        other_bo = deps[1]

        backend.Constant.assign(self.get_outputs()[0].get_saved_output(), other_bo.get_saved_output())

