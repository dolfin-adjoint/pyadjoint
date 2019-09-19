"""
Compatibility layer for ipopt wrappers
Currently only cyipopt is supported

Repos:

- https://github.com/matthias-k/cyipopt
"""

try:
    import cyipopt  # noqa: F401
except ImportError:
    raise ImportError("You need to install cyipopt. It is recommended to install IPOPT with HSL support!")
