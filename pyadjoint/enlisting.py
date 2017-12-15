class Enlist(list):
    def __init__(self, x):
        self.listed = isinstance(x, (list, tuple))
        super(Enlist, self).__init__(x if self.listed else [x])

    def delist(self, y=None):
        y = self if y is None else y
        if self.listed:
            return y
        else:
            assert len(y) == 1
            return y[0]
