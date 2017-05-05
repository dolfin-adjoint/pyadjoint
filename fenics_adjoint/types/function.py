import backend
from pyadjoint.tape import get_working_tape
from pyadjoint.block import Block
from pyadjoint.overloaded_type import OverloadedType


class Function(OverloadedType, backend.Function):
    def __init__(self, *args, **kwargs):
        super(Function, self).__init__(*args, **kwargs)
        backend.Function.__init__(self, *args, **kwargs)

    def copy(self, *args, **kwargs):
        # Overload the copy method so we actually return overloaded types.
        # Otherwise we might end up getting unexpected errors later.
        c = backend.Function.copy(self, *args, **kwargs)
        return Function(c.function_space(), c.vector())

    def assign(self, other, *args, **kwargs):
        annotate_tape = kwargs.pop("annotate_tape", True)
        if annotate_tape:
            block = AssignBlock(self, other)
            tape = get_working_tape()
            tape.add_block(block)

        ret = super(Function, self).assign(other, *args, **kwargs)

        if annotate_tape:
            block.add_output(self.create_block_output())

        return ret

    def get_derivative(self, options={}):
        adj_value = self.get_adj_output()
        riesz_representation = options.pop("riesz_representation", "l2")

        if riesz_representation == "l2":
            func = Function(self.function_space(), adj_value)
            return func
        elif riesz_representation == "L2":
            ret = Function(self.function_space())
            u = backend.TrialFunction(self.function_space())
            v = backend.TestFunction(self.function_space())
            M = backend.assemble(u*v*backend.dx)
            backend.solve(M, ret.vector(), adj_value)
            return ret

    def _ad_create_checkpoint(self):
        return self.copy(deepcopy=True)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def adj_update_value(self, value):
        super(Function, self).assign(value)
        self.original_block_output.save_output()

    def _ad_mul(self, other):
        r = Function(self.function_space())
        backend.Function.assign(r, self*other)
        return r

    def _ad_add(self, other):
        r = Function(self.function_space())
        backend.Function.assign(r, self+other)
        return r

    def _ad_dot(self, other):
        return self.vector().inner(other.vector())


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

    def recompute(self):
        deps = self.get_dependencies()
        other_bo = deps[1]

        backend.Function.assign(self.get_outputs()[0].get_saved_output(), other_bo.get_saved_output())
