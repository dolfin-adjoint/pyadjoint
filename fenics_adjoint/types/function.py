import backend
from . import compat
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.tape import get_working_tape, annotate_tape, stop_annotating, no_annotations
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType, FloatingType
from .compat import gather


class Function(FloatingType, backend.Function):
    def __init__(self, *args, **kwargs):
        super(Function, self).__init__(*args,
                                       block_class=kwargs.pop("block_class", None),
                                       _ad_floating_active=kwargs.pop("_ad_floating_active", False),
                                       _ad_args=kwargs.pop("_ad_args", None),
                                       output_block_class=kwargs.pop("output_block_class", None),
                                       _ad_output_args=kwargs.pop("_ad_output_args", None),
                                       _ad_outputs=kwargs.pop("_ad_outputs", None),
                                       annotate=kwargs.pop("annotate", True),
                                       **kwargs)
        backend.Function.__init__(self, *args, **kwargs)

    def copy(self, *args, **kwargs):
        from .types import create_overloaded_object

        annotate = annotate_tape(kwargs)
        c = backend.Function.copy(self, *args, **kwargs)
        func = create_overloaded_object(c)

        if annotate:
            if kwargs.pop("deepcopy", False):
                block = AssignBlock(func, self)
                tape = get_working_tape()
                tape.add_block(block)
                block.add_output(func.create_block_variable())
            else:
                # TODO: Implement. Here we would need to use floating types.
                pass

        return func

    def assign(self, other, *args, **kwargs):
        '''To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Dolfin assign call.'''
        # do not annotate in case of self assignment
        annotate = annotate_tape(kwargs) and self != other
        if annotate:
            from .types import create_overloaded_object
            other = create_overloaded_object(other)
            block = AssignBlock(self, other)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            ret = super(Function, self).assign(other, *args, **kwargs)

        if annotate:
            block.add_output(self.create_block_variable())

        return ret

    def split(self, *args, **kwargs):
        deepcopy = kwargs.get("deepcopy", False)
        annotate = annotate_tape(kwargs)
        if deepcopy or not annotate:
            # TODO: This is wrong for deepcopy=True. You need to convert every subfunction into an OverloadedType.
            return backend.Function.split(self, *args, **kwargs)

        num_sub_spaces = backend.Function.function_space(self).num_sub_spaces()
        ret = [Function(self, i,
                        block_class=SplitBlock,
                        _ad_floating_active=True,
                        _ad_args=[self, i],
                        _ad_output_args=[i],
                        output_block_class=MergeBlock,
                        _ad_outputs=[self])
               for i in range(num_sub_spaces)]
        return tuple(ret)

    def vector(self):
        vec = backend.Function.vector(self)
        vec.function = self
        return vec

    @no_annotations
    def _ad_convert_type(self, value, options=None):
        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")

        if riesz_representation == "l2":
            return Function(self.function_space(), value)
        elif riesz_representation == "L2":
            ret = Function(self.function_space())
            u = backend.TrialFunction(self.function_space())
            v = backend.TestFunction(self.function_space())
            M = backend.assemble(u * v * backend.dx)
            if not isinstance(value, backend.Vector):
                value = value.vector()
            backend.solve(M, ret.vector(), value)
            return ret

    def _ad_create_checkpoint(self):
        if self.block is None:
            # TODO: This might crash if annotate=False, but still using a sub-function.
            #       Because subfunction.copy(deepcopy=True) raises the can't access vector error.
            return self.copy(deepcopy=True)

        dep = self.block.get_dependencies()[0]
        return backend.Function.sub(dep.saved_output, self.block.idx, deepcopy=True)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    @no_annotations
    def adj_update_value(self, value):
        self.original_block_variable.checkpoint = value._ad_create_checkpoint()

    @no_annotations
    def _ad_mul(self, other):
        r = Function(self.function_space())
        backend.Function.assign(r, self*other)
        return r

    @no_annotations
    def _ad_add(self, other):
        r = Function(self.function_space())
        backend.Function.assign(r, self+other)
        return r

    def _ad_dot(self, other, options=None):
        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return self.vector().inner(other.vector())
        elif riesz_representation == "L2":
            return backend.assemble(self * other * backend.dx)

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
        m_a = gather(m_v)

        return m_a.tolist()

    def _ad_copy(self):
        r = Function(self.function_space())
        backend.Function.assign(r, self)
        return r

    def _ad_dim(self):
        return self.function_space().dim()

    def _imul(self, other):
        vec = self.vector()
        vec *= other

    def _iadd(self, other):
        vec = self.vector()
        vec += other.vector()

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

    def _applyBinary(self, f, y):
        vec = self.vector()
        npdata = vec.get_local()
        npdatay = y.vector().get_local()
        for i in range(len(vec)):
            npdata[i] = f(npdata[i], npdatay[i])
        vec.set_local(npdata)


class AssignBlock(Block):
    def __init__(self, func, other):
        super(AssignBlock, self).__init__()
        self.add_dependency(func.block_variable)
        self.add_dependency(other.block_variable)

    @no_annotations
    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return
        if isinstance(self.get_dependencies()[1], AdjFloat):
            adj_input = adj_input.sum()
        self.get_dependencies()[1].add_adj_output(adj_input)

    @no_annotations
    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[1].tlm_value
        if tlm_input is None:
            return
        self.get_outputs()[0].add_tlm_output(tlm_input)

    @no_annotations
    def evaluate_hessian(self):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        self.get_dependencies()[1].add_hessian_output(hessian_input)

    @no_annotations
    def recompute(self):
        deps = self.get_dependencies()
        other_bo = deps[1]
        # TODO: This is a quick-fix, so should be reviewed later.
        #       Introduced because Control values work by not letting their checkpoint property be overwritten.
        #       However this circumvents that by using assign. An alternative would be to create a
        #       Function, assign the values to the new function, and then set the checkpoint to this function.
        #       However that is rather memory inefficient I would say.
        if not self.get_outputs()[0].is_control:
            backend.Function.assign(self.get_outputs()[0].saved_output, other_bo.saved_output)


class SplitBlock(Block):
    def __init__(self, func, idx):
        super(SplitBlock, self).__init__()
        self.add_dependency(func.block_variable)
        self.idx = idx

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return
        dep = self.get_dependencies()[0]
        dep.add_adj_output(adj_input)

    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        self.get_outputs()[0].add_tlm_output(
            backend.Function.sub(tlm_input, self.idx, deepcopy=True)
        )

    def evaluate_hessian(self):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        dep = self.get_dependencies()[0]
        dep.add_hessian_output(hessian_input)

    def recompute(self):
        dep = self.get_dependencies()[0].checkpoint
        self.get_outputs()[0].checkpoint = backend.Function.sub(dep, self.idx, deepcopy=True)


class MergeBlock(Block):
    def __init__(self, func, idx):
        super(MergeBlock, self).__init__()
        self.add_dependency(func.block_variable)
        self.idx = idx

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return
        dep = self.get_dependencies()[0]
        dep.add_adj_output(adj_input)

    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        output = self.get_outputs()[0]
        fs = output.output.function_space()
        f = backend.Function(fs)
        output.add_tlm_output(
            backend.assign(f.sub(self.idx), tlm_input)
        )

    def evaluate_hessian(self):
        hessian_input = self.get_outputs()[0].hessian_value
        if hessian_input is None:
            return
        dep = self.get_dependencies()[0]
        dep.add_hessian_output(hessian_input)

    def recompute(self):
        dep = self.get_dependencies()[0].checkpoint
        output = self.get_outputs()[0].checkpoint
        backend.assign(backend.Function.sub(output, self.idx), dep)

