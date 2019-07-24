from pyadjoint.overloaded_function import overload_function, overloaded_function
from pyadjoint import Block, AdjFloat, ReducedFunctional, Control
from pyadjoint.tape import no_annotations, Tape, set_working_tape
import numpy as np


class SinBlock(Block):

    def __init__(self, x, **kwargs):
        super(SinBlock, self).__init__(**kwargs)
        self.add_dependency(x)

    @no_annotations
    def evaluate_adj(self, markings=False):
        x = self.get_dependencies()[0].saved_output
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return
        dsin = np.cos(x) * adj_input
        self.get_dependencies()[0].add_adj_output(dsin)


class CosBlock(Block):

    def __init__(self, x, **kwargs):
        super(CosBlock, self).__init__(**kwargs)
        self.add_dependency(x)

    @no_annotations
    def evaluate_adj(self, markings=False):
        x = self.get_dependencies()[0].saved_output
        adj_input = self.get_outputs()[0].adj_value
        if adj_input is None:
            return
        dcos = -np.sin(x) * adj_input
        self.get_dependencies()[0].add_adj_output(dcos)


def ad_sin(y):
    return AdjFloat(np.sin(y))


@overloaded_function(CosBlock)
def cos(x):
    return AdjFloat(np.sin(x))


def test_overload_function():
    tape = Tape()
    set_working_tape(tape)
    sin = overload_function(ad_sin, SinBlock)

    z = AdjFloat(5)
    t = AdjFloat(3)
    r = sin(z)
    q = cos(r * t)

    Jhat = ReducedFunctional(q, Control(z))
    assert(Jhat.derivative() == -np.sin(t * np.sin(z)) * t * np.cos(z))
