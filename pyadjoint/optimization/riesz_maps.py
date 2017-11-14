# Dummy objects that allow for a compact notation in the
# configuration of Riesz maps.

from backend import TrialFunction, TestFunction, grad, inner, dx, assemble, Constant

__all__ = ["BaseRieszMap", "L2", "H10", "H1"]

class BaseRieszMap(object):
    def __init__(self, V):
        self.V = V

class L2(BaseRieszMap):
    def assemble(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        A = inner(u, v)*dx
        return assemble(A)

class H10(BaseRieszMap):
    def assemble(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        A = inner(grad(u), grad(v))*dx
        return assemble(A)

class H1(BaseRieszMap):
    def __init__(self, V, alpha=None):
        BaseRieszMap.__init__(self, V)

        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = Constant(1.0)

    def assemble(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        A = inner(u, v)*dx + alpha*inner(grad(u), grad(v))*dx
        return assemble(A)
