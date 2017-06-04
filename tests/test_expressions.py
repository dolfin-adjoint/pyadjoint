from fenics import *
from fenics_adjoint import *

# For asserts
from pyadjoint.overloaded_type import OverloadedType

def test_subclass_expression():
    class MyExpression1(Expression):
        def eval_cell(self, value, x, ufc_cell):
            if ufc_cell.index > 10:
                value[0] = 1.0
            else:
                value[0] = -1.0

    f = MyExpression1(degree=1)

    # Expression is a base (seems obvious from the definition above,
    # but metaclass magic makes it not so obvious):
    assert(Expression in MyExpression1.__bases__)

    # OverloadedType is a base of the subclass:
    assert(OverloadedType in MyExpression1.__bases__)
    assert(isinstance(f, OverloadedType))


def test_jit_expression():
    f = Expression("a*sin(k*pi*x[0])*cos(k*pi*x[1])", a=2, k=3, degree=2)

    assert(isinstance(f, OverloadedType))


def test_jit_expression_evaluations():
    f = Expression("u", u=1, degree=1)

    assert(f.u == 1)
    assert(f(0.0) == 1)

    f.user_parameters['u'] = 2

    assert(f(0.0) == 2)
    assert(f.u == 2)

def test_ignored_expression_attributes():
    ignored_attrs = []

    class _DummyExpressionClass(Expression):
        def eval(self, value, x):
            pass

    tmp = _DummyExpressionClass(degree=1, annotate_tape=False)
    ignored_attrs += dir(tmp)
    tmp = Expression("1", degree=1, annotate_tape=False)
    ignored_attrs += dir(tmp)

    from sys import version_info
    if version_info.major < 3:
        # Attributes added in python3
        ignored_attrs.append("__dir__")
        ignored_attrs.append("__init_subclass__")
    elif version_info.minor < 6:
        # Attributes added in python3.6
        ignored_attrs.append("__init_subclass__")

    from fenics_adjoint.types.expression import _IGNORED_EXPRESSION_ATTRIBUTES 
    assert(set(ignored_attrs) == set(_IGNORED_EXPRESSION_ATTRIBUTES))

def test_cpp_inline():
    # An expression that depends on a and b
    base_code = '''
    class MyCppExpression : public Expression
    {
    public:
          std::shared_ptr<Constant> a;
          std::shared_ptr<Constant> b;
      MyCppExpression() : Expression() {}

      void eval(Array<double>& values, const Array<double>& x) const
      {
        double a_ = (double) *a;
        double b_ = (double) *b;
        values[0] = EXPRESSION;
      }
    };'''

    cpp_code = base_code.replace("EXPRESSION", "(x[0] - a_)*b_*b_*a_")
    da_cpp_code = base_code.replace("EXPRESSION", "(x[0] - a_)*b_*b_ - b_*b_*a_")
    db_cpp_code = base_code.replace("EXPRESSION", "2*(x[0] - a_)*b_*a_")

    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    a = Constant(0.5)
    b = Constant(0.25)

    def J(a):
        if not isinstance(a, Constant):
            a = Constant(a)
        f = Expression(cpp_code, degree=1)
        f.a = a; f.b = b

        dfda = Expression(da_cpp_code, degree=1)
        dfda.a = a; dfda.b = b

        dfdb = Expression(db_cpp_code, degree=1)
        dfdb.a = a; dfdb.b = b

        f.user_defined_derivatives = {a: dfda, b: dfdb}

        return assemble(f**2*dx(domain=mesh))

    _test_adjoint_constant(J, a)

def test_inline_function_control():
    mesh = IntervalMesh(100, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    g = project(Expression("sin(x[0])", degree=1, annotate_tape=False), V, annotate_tape=False)

    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, Constant(1), "on_boundary")

    def J(g):
        u_ = Function(V)

        f = Expression("g", g=g, degree=1)
        f_deriv = Expression("1", degree=1)
        f.user_defined_derivatives = {g: f_deriv}

        a = inner(grad(u), grad(v))*dx
        L = f*v*dx        

        solve(a == L, u_, bc)
        return assemble(u_**2*dx)

    _test_adjoint(J, g)

class UserDefinedExpr(Expression):
    def __init__(self, m, t, **kwargs):
        self.m = m
        self.t = t

    def eval(self, value, x):
        value[0] = self.m*self.t

def test_time_dependent_class():
    # Defining the domain, 100 points from 0 to 1
    mesh = IntervalMesh(100, 0, 1)

    # Defining function space, test and trial functions
    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    # Marking the boundaries
    def left(x, on_boundary):
        return near(x[0],0)

    def right(x, on_boundary):
        return near(x[0],1)

    # Dirichlet boundary conditions
    bc_left = DirichletBC(V, 1, left)
    bc_right = DirichletBC(V, 2, right)
    bc = [bc_left, bc_right]

    # Some variables
    T = 0.2
    dt = 0.1
    f = Constant(2.0)

    def J(f):
        if not isinstance(f, Constant):
            f = Constant(f)
        u_1 = Function(V)
        u_1.vector()[:] = 1 
        expr = UserDefinedExpr(m=f, t=dt, degree=1)
        expr_deriv = UserDefinedExpr(m=1.0, t=dt, degree=1, annotate_tape=False)
        expr._ad_ignored_attributes = ["m"]
        expr.user_defined_derivatives = {f: expr_deriv}

        a = u_1*u*v*dx + dt*expr*inner(grad(u),grad(v))*dx
        L = u_1*v*dx

        # Time loop
        t = dt
        while t <= T:
            solve(a == L, u_, bc)
            u_1.assign(u_)
            t += dt
            expr.t = t

        return assemble(u_1**2*dx(domain=mesh))

    _test_adjoint_constant(J, f)

def test_time_dependent_inline():
    # Defining the domain, 100 points from 0 to 1
    mesh = IntervalMesh(100, 0, 1)

    # Defining function space, test and trial functions
    V = FunctionSpace(mesh,"CG",1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    # Marking the boundaries
    def left(x, on_boundary):
        return near(x[0],0)

    def right(x, on_boundary):
        return near(x[0],1)

    # Dirichlet boundary conditions
    bc_left = DirichletBC(V, 1, left)
    bc_right = DirichletBC(V, 2, right)
    bc = [bc_left, bc_right]

    # Some variables
    T = 0.2
    dt = 0.1
    f = Constant(2.0)

    def J(f):
        if not isinstance(f, Constant):
            f = Constant(f)
        u_1 = Function(V)
        u_1.vector()[:] = 1 
        expr = Expression("f*t", f=f, t=dt, degree=1)
        expr_deriv = Expression("t", t=dt, degree=1, annotate_tape=False)
        expr.user_defined_derivatives = {f: expr_deriv}

        a = u_1*u*v*dx + dt*expr*inner(grad(u),grad(v))*dx
        L = u_1*v*dx

        # Time loop
        t = dt
        while t <= T:
            solve(a == L, u_, bc)
            u_1.assign(u_)
            t += dt
            expr.t = t

        return assemble(u_1**2*dx(domain=mesh))

    _test_adjoint_constant(J, f)

# Since ReducedFunctional and verification.py (taylor test function)
# doesn't exist in this branch, we use the old way temporary for these tests.
def _test_adjoint_constant(J, c):
    import numpy.random
    tape = Tape()
    set_working_tape(tape)

    h = Constant(1)

    eps_ = [0.01/2.0**i for i in range(4)]
    residuals = []
    for eps in eps_:

        Jp = J(c + eps*h)
        tape.clear_tape()
        Jm = J(c)
        #tape.visualise(dot=True, filename="expr.dot")
        #import sys; sys.exit()
        Jm.set_initial_adj_input(1.0)
        tape.evaluate()

        dJdc = c.get_adj_output()
        print(dJdc)

        residual = abs(Jp - Jm - eps*dJdc)
        residuals.append(residual)

    print(residuals)
    r = convergence_rates(residuals, eps_)
    print(r)

    tol = 1E-1
    assert( r[-1] > 2-tol )

def _test_adjoint(J, f):
    import numpy.random
    tape = Tape()
    set_working_tape(tape)

    V = f.function_space()
    h = Function(V)
    h.vector()[:] = numpy.random.rand(V.dim())
    g = Function(V)

    eps_ = [0.01/2.0**i for i in range(5)]
    residuals = []
    for eps in eps_:
        g.vector()[:] = f.vector()[:] + eps*h.vector()[:]
        Jp = J(g)
        tape.clear_tape()
        Jm = J(f)
        Jm.set_initial_adj_input(1.0)
        tape.evaluate()

        dJdf = f.get_adj_output()

        residual = abs(Jp - Jm - eps*dJdf.inner(h.vector()))
        residuals.append(residual)

    print(residuals)
    r = convergence_rates(residuals, eps_)
    print(r)

    tol = 1E-1
    assert( r[-1] > 2-tol )

def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i]/E_values[i-1])/log(eps_values[i]/eps_values[i-1]))

    return r
