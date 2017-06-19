import sys
if not 'backend' in sys.modules:
    import firedrake
    sys.modules['backend'] = firedrake
else:
    raise ImportError("'backend' module already exists?")

from fenics_adjoint.ui import *
