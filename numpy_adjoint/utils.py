import numpy


def adjoint_broadcast(src, shape):
    """Adjoint of the broadcast from `shape` to `src.shape`

    Args:
        src (numpy.ndarray): an array to adjoint ("reverse") broadcast.
        shape (tuple): a target shape tuple

    Returns:
        numpy.ndarray: an array with shape `shape`, consisting of sums along the forward broadcasted axes.

    """
    src_shape = src.shape

    sum_axes = [i for i in range(-1, -len(shape)-1, -1) if src_shape[i] != shape[i]]
    sum_axes += list(range(len(src_shape) - len(shape)))

    out = numpy.sum(src, axis=tuple(sum_axes), keepdims=True)
    return out.reshape(shape)
