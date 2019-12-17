class ProjectBlock(SolveBlock):
    def __init__(self, v, V, output, bcs=[], *args, **kwargs):
        mesh = kwargs.pop("mesh", None)
        if mesh is None:
            mesh = V.mesh()
        dx = backend.dx(mesh)
        w = backend.TestFunction(V)
        Pv = backend.TrialFunction(V)
        a = backend.inner(w, Pv) * dx
        L = backend.inner(w, v) * dx

        super(ProjectBlock, self).__init__(a == L, output, bcs, *args, **kwargs)
