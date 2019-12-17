from . import SolveBlock


class ProjectBlock(SolveBlock):
    def __init__(self, v, V, output, bcs=[], *args, **kwargs):
        mesh = kwargs.pop("mesh", None)
        if mesh is None:
            mesh = V.mesh()
        dx = self.backend.dx(mesh)
        w = self.backend.TestFunction(V)
        Pv = self.backend.TrialFunction(V)
        a = self.backend.inner(w, Pv) * dx
        L = self.backend.inner(w, v) * dx

        super(ProjectBlock, self).__init__(a == L, output, bcs, *args, **kwargs)
