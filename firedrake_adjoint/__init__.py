# flake8: noqa
from firedrake.adjoint import *

continue_annotation()

raise DeprecationWarning(
"""The firedrake_adjoint module is deprecated.

Instead, use the firedrake.adjoint module and explicitly start taping
by calling continue_annotation()."""
)
