import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *

def test_projection():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    class MyExpression1(UserExpression):
        def __init__(self, c, **kwargs):
            UserExpression.__init__(self, **kwargs)
            self.c = c

        def eval_cell(self, value, x, ufc_cell):
            if ufc_cell.index > 10:
                value[0] = 1.0*self.c
            else:
                value[0] = -1.0*self.c

    class MyExpression1Derivative(UserExpression):
        def __init__(self, **kwargs):
            UserExpression.__init__(self, **kwargs)

        def eval_cell(self, value, x, ufc_cell):
            if ufc_cell.index > 10:
                value[0] = 1.0
            else:
                value[0] = -1.0

    class MyExpression1DerivativeDerivative(UserExpression):
        def __init__(self, **kwargs):
            UserExpression.__init__(self, **kwargs)

        def eval_cell(self, value, x, ufc_cell):
            if ufc_cell.index > 10:
                value[0] = 0.0
            else:
                value[0] = 0.0

    c = AdjFloat(3.0)
    f = MyExpression1(c, degree=1)
    dfdc = MyExpression1Derivative(degree=1)
    f.user_defined_derivatives = {c: dfdc}
    dfdc.user_defined_derivatives = {c: MyExpression1DerivativeDerivative(degree=1)}

    u = project(f, V)

    J = assemble(u**4*dx)
    Jhat = ReducedFunctional(J, Control(c))

    results = taylor_to_dict(Jhat, c, AdjFloat(1.0))

    assert min(results["FD"]["Rate"]) > 0.9
    assert min(results["dJdm"]["Rate"]) > 1.9
    assert min(results["Hm"]["Rate"]) > 2.9


def test_multiple_meshes():
    mesh = UnitSquareMesh(20, 20)
    S1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    Y = FunctionSpace(mesh, S1)

    t = Constant(2.0)
    y_expr = Expression('x[1]*t', t=t, degree=3, name='y')
    y_expr.user_defined_derivatives = {t: Expression('x[1]', degree=3, name='dy')}

    y = project(y_expr, Y)

    mesh_2 = UnitSquareMesh(40, 40)
    Y_2 = FunctionSpace(mesh_2, S1)

    project(y, Y_2)

    J = assemble(y**4*dx)
    Jhat = ReducedFunctional(J, Control(t))

    h = Constant(1)
    results = taylor_to_dict(Jhat, t, h)

    assert min(results["FD"]["Rate"]) > 0.9
    assert min(results["dJdm"]["Rate"]) > 1.9
    assert min(results["Hm"]["Rate"]) > 2.9
