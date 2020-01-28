import backend
from dolfin_adjoint_common import compat
from pyadjoint.tape import get_working_tape, stop_annotating
from pyadjoint.overloaded_type import OverloadedType, create_overloaded_object, register_overloaded_type
from pyadjoint.reduced_functional_numpy import gather

from dolfin_adjoint_common.blocks.constant import constant_from_values

from fenics_adjoint.blocks import ConstantAssignBlock

import numpy

compat = compat.compat(backend)


@register_overloaded_type
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

            block = ConstantAssignBlock(other)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            ret = backend.Constant.assign(self, *args, **kwargs)

        if annotate_tape:
            block.add_output(self.create_block_variable())

        return ret

    def get_derivative(self, options={}):
        return self._ad_convert_type(self.adj_value, options=options)

    def adj_update_value(self, value):
        self.original_block_variable.checkpoint = value._ad_create_checkpoint()

    def _ad_convert_type(self, value, options={}):
        if value is None:
            # TODO: Should the default be 0 constant here or return just None?
            return Constant(numpy.zeros(self.ufl_shape))
        value = gather(value)
        value = compat.constant_function_firedrake_compat(value)
        return constant_from_values(self, value)

    def _ad_function_space(self, mesh):
        element = self.ufl_element()
        fs_element = element.reconstruct(cell=mesh.ufl_cell())
        return backend.FunctionSpace(mesh, fs_element)

    def _ad_create_checkpoint(self):
        return constant_from_values(self)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return constant_from_values(self, self.values() * other)

    def _ad_add(self, other):
        return constant_from_values(self, self.values() + other.values())

    def _ad_dot(self, other, options=None):
        return sum(self.values() * other.values())

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        dst.assign(backend.Constant(numpy.reshape(src[offset:offset + dst.value_size()], dst.ufl_shape)))
        offset += dst.value_size()
        return dst, offset

    @staticmethod
    def _ad_to_list(m):
        a = numpy.zeros(m.value_size())
        p = numpy.zeros(m.value_size())
        m.eval(a, p)
        return a.tolist()

    def _ad_copy(self):
        return constant_from_values(self)

    def _ad_dim(self):
        return numpy.prod(self.values().shape)

    def _ad_imul(self, other):
        self.assign(constant_from_values(self, self.values() * other))

    def _ad_iadd(self, other):
        self.assign(constant_from_values(self, self.values() + other.values()))

    def _reduce(self, r, r0):
        npdata = self.values()
        for i in range(len(npdata)):
            r0 = r(npdata[i], r0)
        return r0

    def _applyUnary(self, f):
        npdata = self.values()
        npdatacopy = npdata.copy()
        for i in range(len(npdata)):
            npdatacopy[i] = f(npdata[i])
        self.assign(constant_from_values(self, npdatacopy))

    def _applyBinary(self, f, y):
        npdata = self.values()
        npdatacopy = self.values().copy()
        npdatay = y.values()
        for i in range(len(npdata)):
            npdatacopy[i] = f(npdata[i], npdatay[i])
        self.assign(constant_from_values(self, npdatacopy))

    def __deepcopy__(self, memodict={}):
        return constant_from_values(self)
