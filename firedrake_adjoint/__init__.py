# flake8: noqa

import warnings

from firedrake.adjoint import *

continue_annotation()

warnings.warn("""The firedrake_adjoint module is deprecated.

Instead, use the firedrake.adjoint module and explicitly start taping
by calling continue_annotation().""", FutureWarning)
