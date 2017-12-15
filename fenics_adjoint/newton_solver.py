import backend
from pyadjoint.tape import annotate_tape, get_working_tape
from .solving import SolveBlock


class NewtonSolver(backend.NewtonSolver):
    def solve(self, *args, **kwargs):
        annotate = annotate_tape(kwargs)

        if annotate:
            tape = get_working_tape()
            factory = args[0]
            vec = args[1]
            b = backend.as_backend_type(vec).__class__()

            factory.F(b=b, x=vec)

            F = b.form
            bcs = b.bcs

            u = vec.function

            block = SolveBlock(F == 0, u, bcs, solver_parameters={"newton_solver": self.parameters.to_dict()})
            tape.add_block(block)

        newargs = [self] + list(args)
        out = backend.NewtonSolver.solve(*newargs, **kwargs)

        if annotate:
            block.add_output(u.create_block_variable())

        return out
