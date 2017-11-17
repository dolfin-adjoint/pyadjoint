''' A simple test that compares the functional value computed manually and with libadjoints functional_evaluation.
    Writting this test was motivated by the bug described on https://bugs.launchpad.net/dolfin-adjoint/+bug/1032291 '''
from __future__ import print_function
from fenics import *
from fenics_adjoint import *
import sys

# Global settings
set_log_level(ERROR)

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "DG", 1)

u_new = Function(V, name = "u_new")
u_old = Function(V, name = "u_old")
m = Control(u_old)
u_test = TestFunction(V)

T = 2.
t = 0.
dlt = 1.
F1 = ( inner((u_new - u_old)/dlt, u_test)*dx - inner(Constant(1.), u_test)*dx )
#solve(inner(u_new, u_test)*dx == 0, u_new)

man_func_value = 0.
print("+++++++++++++ INITIAL RUN +++++++++")
man_func_value_contr = 0.5*assemble(inner(u_new, u_new)*dx)
while t < T:

    solve(F1 == 0, u_new)
    u_old.assign(u_new)

    t += dlt
    man_func_value_contr = assemble(inner(u_new, u_new)*dx)
    if t>=T:
        man_func_value += 0.5*man_func_value_contr
    else:
        man_func_value += man_func_value_contr

print()
print("+++++++++++++ REPLAY +++++++++")
u_new.vector()[:] = 0.
u_old.vector()[:] = 0.
J = man_func_value
reduced_functional = ReducedFunctional(J, m)
reduced_functional_value = reduced_functional(u_old)

assert abs(reduced_functional_value - man_func_value) < 1e-13
