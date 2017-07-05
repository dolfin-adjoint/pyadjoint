from fenics import *
from fenics_adjoint import *
from pyadjoint.block import Block
from pyadjoint.tape import annotate_tape, stop_annotating
from fenics_adjoint.types import create_overloaded_object
from normalise import normalise

backend_normalise = normalise
def normalise(func, **kwargs):
    annotate = annotate_tape(kwargs)

    if annotate:
        tape = get_working_tape()
        block = NormaliseBlock(func)
        tape.add_block(block)

    with stop_annotating():
        output = backend_normalise(func,**kwargs)

    output = create_overloaded_object(output)

    if annotate:
        block.add_output(output.create_block_output())

    return output

class NormaliseBlock(Block):
    def __init__(self, func, **kwargs):
        super(NormaliseBlock,self).__init__()
        self.add_dependency(func.get_block_output())
        self.kwargs = kwargs

    def __str__(self):
        return 'NormaliseBlock'

    def evaluate_adj(self):
        adj_input = self.get_outputs()[0].adj_value
        dependency = self.get_dependencies()[0]
        x = dependency.get_saved_output().vector()

        adj_output = x.copy()

        xnorm = x.norm('l2')

        const = 0
        for i in range(len(x)):
            const += adj_input[i][0]*x[i][0]
        const /= xnorm**3

        for i in range(len(x)):
            adj_output[i] = adj_input[i][0]/xnorm - const*x[i][0]
        dependency.add_adj_output(adj_output)


    def recompute(self):
        dependencies = self.get_dependencies()
        func = dependencies[0].get_saved_output()
        output = backend_normalise(func,**self.kwargs)
        self.get_outputs()[0].checkpoint = output
