# flake8: noqa

import pyadjoint
__version__ = pyadjoint.__version__
__author__ = 'Sebastian Kenji Mitusch'
__credits__ = []
__license__ = 'LGPL-3'
__maintainer__ = 'Sebastian Kenji Mitusch'
__email__ = 'sebastkm@simula.no'


from .array import ndarray

# Use pyadjoint AdjFloat for numpy.float64.
import numpy
from pyadjoint.overloaded_type import register_overloaded_type
from pyadjoint.adjfloat import AdjFloat
register_overloaded_type(AdjFloat, numpy.float64)
