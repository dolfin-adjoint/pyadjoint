import pyadjoint
__version__ = pyadjoint.__version__

import sys
if not 'backend' in sys.modules:
    import firedrake
    sys.modules['backend'] = firedrake
else:
    raise ImportError("'backend' module already exists?")

from firedrake_adjoint.ui import *
