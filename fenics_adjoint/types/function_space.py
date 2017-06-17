import backend


class FunctionSpace(backend.FunctionSpace):
    def sub(self, i):
        V = backend.FunctionSpace.sub(self, i)
        V._ad_parent_space = self
        return V


def extract_subfunction(u, V):
    component = V.component()
    r = u
    for idx in component:
        r = r.sub(int(idx))
    return r
