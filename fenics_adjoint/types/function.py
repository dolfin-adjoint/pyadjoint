import backend
import ufl

from pyadjoint.overloaded_type import (FloatingType,
                                       create_overloaded_object,
                                       register_overloaded_type,
                                       get_overloaded_class)
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating, \
    no_annotations

from dolfin_adjoint_common import compat

from pyadjoint.enlisting import Enlist
import numpy
from fenics_adjoint.blocks import (FunctionEvalBlock, FunctionMergeBlock,
                                   FunctionSplitBlock, FunctionAssignBlock)

compat = compat.compat(backend)


@register_overloaded_type
class Function(FloatingType, backend.Function):
    def __init__(self, *args, **kwargs):
        super(Function, self).__init__(*args,
                                       block_class=kwargs.pop("block_class",
                                                              None),
                                       _ad_floating_active=kwargs.pop(
                                           "_ad_floating_active", False),
                                       _ad_args=kwargs.pop("_ad_args", None),
                                       output_block_class=kwargs.pop(
                                           "output_block_class", None),
                                       _ad_output_args=kwargs.pop(
                                           "_ad_output_args", None),
                                       _ad_outputs=kwargs.pop("_ad_outputs",
                                                              None),
                                       annotate=kwargs.pop("annotate", True),
                                       **kwargs)
        backend.Function.__init__(self, *args, **kwargs)

    @classmethod
    def _ad_init_object(cls, obj):
        return compat.type_cast_function(obj, cls)

    def copy(self, *args, **kwargs):
        annotate = annotate_tape(kwargs)
        c = backend.Function.copy(self, *args, **kwargs)
        func = create_overloaded_object(c)

        if annotate:
            if kwargs.pop("deepcopy", False):
                block = FunctionAssignBlock(func, self)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(func.create_block_variable())
            else:
                # TODO: Implement. Here we would need to use floating types.
                pass

        return func

    def assign(self, other, *args, **kwargs):
        """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin assign call."""
        # do not annotate in case of self assignment
        annotate = annotate_tape(kwargs) and self != other
        if annotate:
            if not isinstance(other, ufl.core.operator.Operator):
                other = create_overloaded_object(other)
            block = FunctionAssignBlock(self, other)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            ret = super(Function, self).assign(other, *args, **kwargs)

        if annotate:
            block.add_output(self.create_block_variable())

        return ret

    def sub(self, i, deepcopy=False, **kwargs):
        from .function_assigner import FunctionAssigner, FunctionAssignerBlock
        annotate = annotate_tape(kwargs)
        if deepcopy:
            ret = create_overloaded_object(backend.Function.sub(self, i, deepcopy, **kwargs))
            if annotate:
                fa = FunctionAssigner(ret.function_space(), self.function_space())
                block = FunctionAssignerBlock(fa, Enlist(self))
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(ret.block_variable)
        else:
            extra_kwargs = {}
            if annotate:
                extra_kwargs = {
                    "block_class": FunctionSplitBlock,
                    "_ad_floating_active": True,
                    "_ad_args": [self, i],
                    "_ad_output_args": [i],
                    "output_block_class": FunctionMergeBlock,
                    "_ad_outputs": [self],
                }
            ret = compat.create_function(self, i, **extra_kwargs)
        return ret

    def split(self, deepcopy=False, **kwargs):
        from .function_assigner import FunctionAssigner, FunctionAssignerBlock
        annotate = annotate_tape(kwargs)
        num_sub_spaces = backend.Function.function_space(self).num_sub_spaces()
        if not annotate:
            if deepcopy:
                ret = tuple(create_overloaded_object(backend.Function.sub(self, i, deepcopy, **kwargs))
                            for i in range(num_sub_spaces))
            else:
                ret = tuple(compat.create_function(self, i)
                            for i in range(num_sub_spaces))
        elif deepcopy:
            ret = []
            fs = []
            for i in range(num_sub_spaces):
                f = create_overloaded_object(backend.Function.sub(self, i, deepcopy, **kwargs))
                fs.append(f.function_space())
                ret.append(f)
            fa = FunctionAssigner(fs, self.function_space())
            block = FunctionAssignerBlock(fa, Enlist(self))
            tape = get_working_tape()
            tape.add_block(block)
            for output in ret:
                block.add_output(output.block_variable)
            ret = tuple(ret)
        else:
            ret = tuple(compat.create_function(self,
                                               i,
                                               block_class=FunctionSplitBlock,
                                               _ad_floating_active=True,
                                               _ad_args=[self, i],
                                               _ad_output_args=[i],
                                               output_block_class=FunctionMergeBlock,
                                               _ad_outputs=[self])
                        for i in range(num_sub_spaces))
        return ret

    def __call__(self, *args, **kwargs):
        annotate = False
        if len(args) == 1 and isinstance(args[0], (numpy.ndarray,)):
            annotate = annotate_tape(kwargs)

        if annotate:
            block = FunctionEvalBlock(self, args[0])
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            out = backend.Function.__call__(self, *args, **kwargs)

        if annotate:
            out = create_overloaded_object(out)
            block.add_output(out.create_block_variable())

        return out

    def vector(self):
        vec = backend.Function.vector(self)
        vec.function = self
        return vec

    @no_annotations
    def _ad_convert_type(self, value, options=None):
        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")

        if riesz_representation == "l2":
            return create_overloaded_object(
                compat.function_from_vector(self.function_space(), value, cls=backend.Function)
            )
        elif riesz_representation == "L2":
            ret = compat.create_function(self.function_space())
            u = backend.TrialFunction(self.function_space())
            v = backend.TestFunction(self.function_space())
            M = backend.assemble(backend.inner(u, v) * backend.dx)
            compat.linalg_solve(M, ret.vector(), value)
            return ret
        elif riesz_representation == "H1":
            ret = compat.create_function(self.function_space())
            u = backend.TrialFunction(self.function_space())
            v = backend.TestFunction(self.function_space())
            M = backend.assemble(
                backend.inner(u, v) * backend.dx + backend.inner(
                    backend.grad(u), backend.grad(v)) * backend.dx)
            compat.linalg_solve(M, ret.vector(), value)
            return ret
        elif callable(riesz_representation):
            return riesz_representation(value)
        else:
            raise NotImplementedError(
                "Unknown Riesz representation %s" % riesz_representation)

    @no_annotations
    def _ad_create_checkpoint(self):
        if self.block is None:
            # TODO: This might crash if annotate=False, but still using a sub-function.
            #       Because subfunction.copy(deepcopy=True) raises the can't access vector error.
            return self.copy(deepcopy=True)

        dep = self.block.get_dependencies()[0].saved_output
        return dep.sub(self.block.idx, deepcopy=False)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    @no_annotations
    def adj_update_value(self, value):
        self.original_block_variable.checkpoint = value._ad_create_checkpoint()

    @no_annotations
    def _ad_mul(self, other):
        r = get_overloaded_class(backend.Function)(self.function_space())
        backend.Function.assign(r, self * other)
        return r

    @no_annotations
    def _ad_add(self, other):
        r = get_overloaded_class(backend.Function)(self.function_space())
        backend.Function.assign(r, self + other)
        return r

    def _ad_dot(self, other, options=None):
        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return self.vector().inner(other.vector())
        elif riesz_representation == "L2":
            return backend.assemble(backend.inner(self, other) * backend.dx)
        elif riesz_representation == "H1":
            return backend.assemble(
                (backend.inner(self, other) + backend.inner(backend.grad(self),
                                                            backend.grad(
                                                                other))) * backend.dx)
        else:
            raise NotImplementedError(
                "Unknown Riesz representation %s" % riesz_representation)

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        range_begin, range_end = dst.vector().local_range()
        m_a_local = src[offset + range_begin:offset + range_end]
        dst.vector().set_local(m_a_local)
        dst.vector().apply('insert')
        offset += dst.vector().size()
        return dst, offset

    @staticmethod
    def _ad_to_list(m):
        if not hasattr(m, "gather"):
            m_v = m.vector()
        else:
            m_v = m
        m_a = compat.gather(m_v)

        return m_a.tolist()

    def _ad_copy(self):
        r = get_overloaded_class(backend.Function)(self.function_space())
        backend.Function.assign(r, self)
        return r

    def _ad_dim(self):
        return self.function_space().dim()

    def _ad_imul(self, other):
        vec = self.vector()
        vec *= other

    def _ad_iadd(self, other):
        vec = self.vector()
        # FIXME: PETSc complains when we add the same vector to itself.
        # So we make a copy.
        vec += other.vector().copy()

    def _reduce(self, r, r0):
        vec = self.vector().get_local()
        for i in range(len(vec)):
            r0 = r(vec[i], r0)
        return r0

    def _applyUnary(self, f):
        vec = self.vector()
        npdata = vec.get_local()
        for i in range(len(npdata)):
            npdata[i] = f(npdata[i])
        vec.set_local(npdata)
        vec.apply("insert")

    def _applyBinary(self, f, y):
        vec = self.vector()
        npdata = vec.get_local()
        npdatay = y.vector().get_local()
        for i in range(len(npdata)):
            npdata[i] = f(npdata[i], npdatay[i])
        vec.set_local(npdata)
        vec.apply("insert")

    def __deepcopy__(self, memodict={}):
        return self.copy(deepcopy=True)
