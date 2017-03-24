from pyadjoint.tape import OverloadedType, Block, get_working_tape

"""
TODO: This may very well be defined in some kind of numpy_adjoint package.
Actually, if this is defined in numpy_adjoint we get a problem with assembly.py
because assemble() can return a float, which needs to be converted to an OverloadedType
by using `create_overloaded_object`.

However we obviously might need an AdjFloat in numpy_adjoint as well.

So the question is what do we do when an adjoint package has a type that is shared by others?
Maybe float is an edge case and since it is supported natively by python we might just
include it in the underlying pyadjoint package.

"""


class AdjFloat(OverloadedType, float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, *args)

    def __init__(self, *args, **kwargs):
        super(AdjFloat, self).__init__(*args, **kwargs)
        float.__init__(self, *args, **kwargs)

    def __mul__(self, other):
        output = float.__mul__(self, other)
        if output is NotImplemented:
            return NotImplemented

        block = MulBlock(self, other)

        tape = get_working_tape()
        tape.add_block(block)

        output = AdjFloat(output)
        block.add_output(output.get_block_output())
        
        return output 


class MulBlock(Block):
    def __init__(self, lfactor, rfactor):
        super(MulBlock, self).__init__()
        self.lfactor = lfactor
        self.rfactor = rfactor

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].get_adj_output()

        self.rfactor.add_adj_output(adj_input * self.lfactor)
        self.lfactor.add_adj_output(adj_input * self.rfactor)

