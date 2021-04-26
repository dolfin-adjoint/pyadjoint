import pytest
pytest.importorskip("firedrake")

from firedrake import *
from firedrake_adjoint import *
from numpy.testing import assert_approx_equal
try:
    import torch
except:
    raise ImportError("Try: pip install torch")


@pytest.fixture(scope='module')
def mesh():
    return IntervalMesh(10, 0, 1)


class PointexprActionOperator(PointexprOperator):

    #def __init__(self, *args, **kwargs):
    #    PointexprOperator.__init__(self, *args, **kwargs)
    def __init__(self, *operands, function_space, derivatives=None, val=None, name=None, coefficient=None, arguments=(), dtype=ScalarType, operator_data):

        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, val=val, name=name, coefficient=coefficient, arguments=arguments, dtype=dtype, operator_data=operator_data)

        # Check
        if not isinstance(operator_data, types.FunctionType):
            error("Expecting a FunctionType pointwise expression")
        expr_shape = operator_data(*operands).ufl_shape
        if expr_shape != function_space.ufl_element().value_shape():
            error("The dimension does not match with the dimension of the function space %s" % function_space)

    def _evaluate_action(self, args):
        if len(args) == 0:
            # Evaluate the operator
            return self._evaluate()

        # Evaluate the Jacobian/Hessian action
        operands = self.ufl_operands
        operator = self._compute_derivatives()
        expr = as_ufl(operator(*operands))
        if expr.ufl_shape == () and expr != 0:
            var = VariableRuleset(self.ufl_operands[0])
            expr = expr*var._Id
        elif expr == 0:
            return self.assign(expr)

        for arg in args:
            mi = indices(len(expr.ufl_shape))
            aa = mi
            bb = mi[-len(arg.ufl_shape):]
            expr = arg[bb] * expr[aa]
            mi_tensor = tuple(e for e in mi if not (e in aa and e in bb))
            if len(expr.ufl_free_indices):
                expr = as_tensor(expr, mi_tensor)
        return self.interpolate(expr)

    def _evaluate_adjoint_action(self, x):
        return self._evaluate_action(x)


def action_point_expr(point_expr, function_space):
    return partial(PointexprActionOperator, operator_data=point_expr, function_space=function_space)


model = torch.nn.Linear(1, 1)
point_op_list = [point_expr, action_point_expr, point_solve, neuralnet]
params_list = [{'operator_data': lambda x:x, 'kwargs': {}},
               {'operator_data': lambda x:x, 'kwargs': {}},
               {'operator_data': lambda x, y: x - y, 'kwargs': {'solver_params':{'tol':1e-7, 'maxiter':30}}},
               {'operator_data': model, 'kwargs': {'inputs_format':1}}]
solver_kwargs = {"mat_type": "matfree"}


@pytest.mark.parametrize(('point_op', 'params'), tuple(zip(point_op_list, params_list)))
def test_solve(point_op, params, mesh):
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    c = Constant(2)

    arg = params['operator_data']
    kwargs = params['kwargs']
    solver_parameters = {} if point_op != action_point_expr else solver_kwargs
    p = point_op(arg, function_space=V, **kwargs)

    """The ExternalOperator is function of the state only: `N(u)`"""
    p2 = p(u)
    def J_u(f):
        F = (-inner(p2, v) + inner(grad(u), grad(v)))*dx - f*v*dx
        solve(F == 0, u, solver_parameters=solver_parameters)
        return assemble(u**2*dx)

    _test_taylor(J_u, f, V)

    """The ExternalOperator is function of the control only: `N(m)`"""
    def J_f(f):
        p2 = p(f)
        F = (-inner(u, v) + inner(grad(u), grad(v)))*dx - p2*v*dx
        solve(F == 0, u, solver_parameters=solver_parameters)
        return assemble(u**2*dx)

    _test_taylor(J_f, f, V)

    if point_op != neuralnet:
        """The ExternalOperator is function of the state and the control: `N(u, m)`"""
        g = point_expr(lambda x,y: x + y, function_space=V)
        if point_op == point_solve:
            g = point_solve(lambda x,y,z: x - y - z, function_space=V, **kwargs)
        def J_u_f(f):
            g2 = g(u,f)
            F = (-inner(g2, v) + inner(grad(u), grad(v)))*dx
            solve(F == 0, u, solver_parameters=solver_parameters)
            return assemble(u**2*dx)

        _test_taylor(J_u_f, f, V)


@pytest.mark.parametrize(('point_op', 'params'), tuple(zip(point_op_list, params_list)))
def test_assemble(point_op, params, mesh):
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    c = Constant(2)

    arg = params['operator_data']
    kwargs = params['kwargs']
    print(point_op, arg, kwargs)
    p = point_op(arg, function_space=V, **kwargs)

    """The ExternalOperator is function of the state only: `N(u)`"""
    p2 = p(u)
    def J_u(f):
        F = (-inner(u, v) + inner(grad(u), grad(v)))*dx - f*v*dx
        solve(F == 0, u)
        return assemble(0.5*u**2*dx + 0.5*p2**2*dx)

    _test_taylor(J_u, f, V)


    """The ExternalOperator is function of the control only: `N(m)`"""
    def J_f(f):
        p2 = p(f)
        F = (-inner(u, v) + inner(grad(u), grad(v)))*dx - f*v*dx
        solve(F == 0, u)
        return assemble(u**2*dx + p2**2*dx)

    _test_taylor(J_f, f, V)


    if point_op != neuralnet:
        """The ExternalOperator is function of the state and the control: `N(u, m)`"""
        g = point_expr(lambda x,y: x + y, function_space=V)
        if point_op == point_solve:
            g = point_solve(lambda x,y,z: x - y - z, function_space=V, **kwargs)
        def J_u_f(f):
            g2 = g(u,f)
            F = (-inner(u, v) + inner(grad(u), grad(v)))*dx - f*v*dx
            solve(F == 0, u)
            return assemble(0.5*u**2*dx + 0.5*g2**2*dx)

        _test_taylor(J_u_f, f, V)


def test_weights_optimization():
    mesh = UnitSquareMesh(5, 5, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 1

    u = Function(V)
    v = TestFunction(V)
    model = torch.nn.Linear(1, 1)

    p = neuralnet(model, function_space=V, inputs_format=1)
    p2 = p(f)

    # Get model parameters (weight and bias)
    m, bias = p2.operator_params()

    def J(m):
        # If last argument is a Constant we don't automatically add model params in the operands list
        p2 = p(f, m, bias)
        F = (-inner(u, v) + inner(grad(u), grad(v)))*dx - p2*v*dx
        solve(F == 0, u)
        return assemble(u**2*dx)

    val = J(m)
    control = Control(m)
    rf = ReducedFunctional(val, control)
    m_opt = minimize(rf, method="L-BFGS-B", tol=1.0e-12, options={"disp": True, "gtol": 1.0e-12, "maxiter" : 20})

    # Minimum reached when u = 0, that is when m = -bias (since we only have 1 Linear layer)!
    assert_approx_equal(m_opt.dat.data_ro, -bias.dat.data_ro)


def _test_taylor(J, f, V):
    h = Function(V)
    h.assign(1, annotate=False)
    val = J(f)
    rf = ReducedFunctional(val, Control(f))
    assert taylor_test(rf, f, h) > 1.9
