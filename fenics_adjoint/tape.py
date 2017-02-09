import backend


_working_tape = None

def get_working_tape():
    return _working_tape

def set_working_tape(tape):
    global _working_tape
    _working_tape = tape

class Tape(object):

    __slots__ = ["blocks"]

    def __init__(self):
        # Initialize the list of blocks on the tape.
        self.blocks = []

    def clear_tape(self):
        self.reset_variables()
        self.blocks = []

    def add_block(self, block):
        """
        Adds a block to the tape and returns the index.
        """
        self.blocks.append(block)

        # len() is computed in constant time, so this should be fine.
        return len(self.blocks)-1

    def evaluate(self):
        for i in range(len(self.blocks)-1, -1, -1):
            self.blocks[i].evaluate_adj()

    def reset_variables(self):
        for i in range(len(self.blocks)-1, -1, -1):
            self.blocks[i].reset_variables()


class Block(object):
    """Base class for all Tape Block types.
    
    Each instance of a Block type represents an elementary operation in the forward model.
    
    Abstract methods:
        evaluate_adj

    Attributes:
        dependencies (set) : a set containing the inputs in the forward model
        fwd_outputs (list) : a list of outputs in the forward model

    """
    def __init__(self):
        self.dependencies = set()
        self.fwd_outputs = []

    def add_dependency(self, dep):
        self.dependencies.add(dep)

    def get_dependencies(self):
        return self.dependencies

    def create_fwd_output(self, obj):
        self.fwd_outputs.append(obj)

    def reset_variables(self):
        for dep in self.dependencies:
            dep.reset_variables()

    def create_reference_object(self, output):
        if isinstance(output, float):
            cls = AdjFloat
        elif isinstance(output, backend.Function):
            cls = Function
        else:
            return NotImplemented

        ret = cls(output)
        self.create_fwd_output(ret)

        return ret

    def evaluate_adj():
        return NotImplemented

class OverloadedType(object):
    def __init__(self, *args, **kwargs):
        tape = kwargs.pop("tape", None)

        if tape:
            self.tape = tape
        else:
            self.tape = get_working_tape()

        self.adj_value = 0

        self._init_(*args, **kwargs)

    def _init_(self):
        # TODO: Remove this method and use super instead.
        return NotImplemented

    def add_adj_output(self, val):
        self.adj_value += val

    def get_adj_output(self):
        return self.adj_value

    def set_initial_adj_input(self, value):
        self.adj_value = value

    def reset_variables(self):
        self.adj_value = 0


class Function(OverloadedType, backend.Function):
    def _init_(self, *args, **kwargs):
        backend.Function.__init__(self, *args, **kwargs)

class Constant(OverloadedType, backend.Constant):
    def _init_(self, *args, **kwargs):
        backend.Constant.__init__(self, *args, **kwargs)

class AdjFloat(OverloadedType, float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, *args)

    def _init_(self, *args, **kwargs):
        float.__init__(self, *args, **kwargs)

    def __mul__(self, other):
        output = float.__mul__(self, other)
        if output is NotImplemented:
            return NotImplemented

        block = AdjFloat.MulBlock(self.tape, self, other)

        output = block.create_reference_type(output)

        return output 


    class MulBlock(Block):
        def __init__(self, lfactor, rfactor):
            super(MulBlock, self).__init__()
            self.lfactor = lfactor
            self.rfactor = rfactor

        def evaluate_adj(self):
            adj_input = self.fwd_outputs[0].get_adj_output()

            self.rfactor.add_adj_output(adj_input * self.lfactor)
            self.lfactor.add_adj_output(adj_input * self.rfactor)