from dolfin import *
from dolfin_adjoint import *
set_log_level(WARNING)

parameters["adjoint"]["debug_cache"] = True

class BidomainSolver(object):
    def __init__(self, dt):
        mesh = UnitIntervalMesh(2)
        V = FunctionSpace(mesh, "CG", 1)

        self.v = TrialFunction(V)
        self.w = TestFunction(V)
        self.s = Function(V)

        self._lhs = dt*self.v*self.w*dx
        self._rhs = self.w*dx

    def step(self):

        mat = assemble(self._lhs)
        solver = LUSolver(mat)
        solver.parameters["reuse_factorization"] = False

        rhs = assemble(self._rhs)

        solver.solve(self.s.vector(), rhs)


dt = Constant(0.1)
solver = BidomainSolver(dt)

solver.step()
dt.assign(0.5)
solver.step()

success = replay_dolfin(tol=0.0, stop=True)
assert success
