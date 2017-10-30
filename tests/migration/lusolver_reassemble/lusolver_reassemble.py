from dolfin import *
from dolfin_adjoint import *

set_log_level(WARNING)
mesh = UnitIntervalMesh(5)


class Solver(object):

    def __init__(self):
        R = FunctionSpace(mesh, "R", 0)
        self.timestep = Function(R)
        self.r = TestFunction(R)

    def step(self, dt):
        V = FunctionSpace(mesh, "CG", 1)
        v = TrialFunction(V)
        w = TestFunction(V)
        s = Function(V)
        r = self.r

        timestep_update = (self.timestep - Constant(dt))*r*dx
        solve(timestep_update == 0, self.timestep)

        G = self.timestep*v*w*dx - w*dx

        lhs, rhs = system(G)
        lhs_matrix = assemble(lhs)
        solver = LUSolver(lhs_matrix)
        solver.parameters["reuse_factorization"] = True

        rhs = assemble(rhs)
        solver.solve(s.vector(), rhs)


solver = Solver()

solver.step(0.1)
solver.step(0.2)

assert replay_dolfin(tol=0.0, stop=True)
