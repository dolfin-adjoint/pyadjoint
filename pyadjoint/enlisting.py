def enlist(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


def delist(x):
    if len(x) == 1:
        return x[0]
    else:
        return x