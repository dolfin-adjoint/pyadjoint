import backend


class FunctionSpace(backend.FunctionSpace):
    def sub(self, i):
        V = backend.FunctionSpace.sub(self, i)
        V._ad_parent_space = self
        return V
