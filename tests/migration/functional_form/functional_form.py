"Test expected failures"

from dolfin import *
from dolfin_adjoint import *

# Define discrete Functionspace
mesh = UnitSquareMesh(20, 20)
V = FunctionSpace(mesh, "CG", 1)

# Define Functions
u = TrialFunction(V)
v = TestFunction(V)
s = Function(V)                   # PDE solution
lmbd = Function(V)                # Adjoint PDE solution
f = Function(V)                   # Parameter
alpha = Constant(1e-6)            # Regularisation parameter
ud = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=1)   # Desired temperature profile

form_valid = 0.5*inner(s-ud, s-ud)*dx + alpha*f*f*dx
J = Functional(form_valid)

# Form has no consistent rank
form_invalid = inner(u-ud, s)*dx
try:
    J = Functional(form_invalid)
except Exception:
    pass

# Form has not rank 0
form_invalid = inner(u, s)*dx
try:
    J = Functional(form_invalid)
except Exception:
    pass

# Form has not rank 0
form_invalid = inner(u, v)*dx
try:
    J = Functional(form_invalid)
except Exception:
    pass
