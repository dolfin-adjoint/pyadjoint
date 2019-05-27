import numpy as np
import numpy_adjoint as npa
from numpy.random import rand

from pyadjoint import *


def test_simple_indexing():
    set_working_tape(Tape())
    arr = npa.array(rand(3))
    J = arr[0]*arr[0] + arr[1]**2 + arr[2]*arr[2]

    Jhat = ReducedFunctional(J, Control(arr))

    assert Jhat(arr) == J

    h = npa.array(rand(3))
    assert taylor_test(Jhat, arr, h) > 1.9


def test_simple_slicing():
    set_working_tape(Tape())
    arr = npa.array(rand(3))
    J = (arr[0:1]*arr[0:1] + arr[1:-1]**2 + arr[2:]*arr[2:])/(arr[0:1] - arr[1:-1])

    Jhat = ReducedFunctional(J, Control(arr))
    assert Jhat(arr) == J

    tape = get_working_tape()
    tape.visualise("test.dot")

    h = npa.array(rand(3))
    assert taylor_test(Jhat, arr, h) > 1.9


if __name__ == "__main__":
    test_simple_slicing()


