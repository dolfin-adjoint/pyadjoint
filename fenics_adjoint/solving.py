import backend
import ufl
from .tape import Tape, Block, Function, get_working_tape

def solve(eq, func, bc):
    output = backend.solve(eq, func, bc)

    tape = get_working_tape()
    block = SolveBlock(eq, func, bc)
    tape.add_block(block)

    return output


class SolveBlock(Block):
    def __init__(self, eq, func, bc):
        super(SolveBlock, self).__init__()
        self.eq = eq
        self.bc = bc
        self.func = func
        if isinstance(self.eq.lhs, ufl.Form) and isinstance(self.eq.rhs, ufl.Form):
            self.linear = True
            # Add dependence on coefficients on the right hand side.
            for c in self.eq.rhs.coefficients():
                self.add_dependency(c)

            # Add solution function to dependencies.
            self.add_dependency(func)
        else:
            self.linear = False

        for c in self.eq.lhs.coefficients():
                self.add_dependency(c)

    def evaluate_adj(self):
        adj_var = Function(self.func.function_space())

        # Obtain (dFdu)^T.
        if self.linear:
            dFdu = self.eq.lhs
        else:
            dFdu = backend.derivative(self.eq.lhs, self.func, backend.TrialFunction(self.func.function_space()))

        adj_dFdu = backend.adjoint(dFdu)
        adj_dFdu = backend.assemble(dFdu)

        # Get dJdu from previous calculations.
        dJdu = self.func.get_adj_output()

        # Homogenize and apply boundary conditions on adj_dFdu and dJdu.
        bc = backend.DirichletBC(self.bc) # TODO: Make it possible to use non-Dirichlet BC/mixed BC.
        bc.homogenize()
        bc.apply(adj_dFdu)
        bc.apply(dJdu)

        # Solve the adjoint equations.
        backend.solve(adj_dFdu, adj_var.vector(), dJdu)

        for c in self.get_dependencies():
            if c != self.func:
                dFdm = -backend.derivative(self.eq.lhs-self.eq.rhs, c, backend.TrialFunction(c.function_space()))
                dFdm = backend.adjoint(dFdm)
                c.add_adj_output(backend.assemble(dFdm)*adj_var.vector())
