from fenics import *
from fenics_adjoint import *


def normalise(func):
    vec = func.vector()
    normalised_vec = vec / vec.norm('l2')
    return Function(func.function_space(), normalised_vec)
