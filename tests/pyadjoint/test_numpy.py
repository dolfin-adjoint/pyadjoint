import numpy as np
from pyadjoint import *
from numpy_adjoint import *


def test_ndarray_getitem_single():
    a = create_overloaded_object(np.array([-2.0]))
    J = ReducedFunctional(a[0], Control(a))
    dJ = J.derivative()
    assert dJ == 1.0
