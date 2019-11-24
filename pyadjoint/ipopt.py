"""
Compatibility layer for pyipopt and its currently-maintained fork ipyopt

Repos:

- https://github.com/xuy/pyipopt
- https://github.com/g-braeunlich/IPyOpt
"""

try:
    from pyipopt import *  # noqa
except ImportError:
    # fallback on ipyopt fork
    try:
        from ipyopt import *  # noqa
    except ImportError:
        raise ImportError("pyadjoint ipopt support requires pyipopt or ipyopt")
