from fenics import *
from fenics_adjoint import *


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


if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    a = Constant(0.5)
    b = Constant(0.25)

    f = Expression(cpp_code, degree=1)
    f.a = a; f.b = b
    f.dependencies = [a, b]

    dfda = Expression(da_cpp_code, degree=1)
    dfda.a = a; dfda.b = b

    dfdb = Expression(db_cpp_code, degree=1)
    dfdb.a = a; dfdb.b = b

    f.user_defined_derivatives = {a: dfda, b: dfdb}

    J = assemble(f**2*dx(domain=mesh))
    rf1 = ReducedFunctional(J, Control(a))
    rf2 = ReducedFunctional(J, Control(b))

    h = Constant(1.0)
    assert taylor_test(rf1, a, h) > 1.9
    assert taylor_test(rf2, b, h) > 1.9

    rf3 = ReducedFunctional(J, [Control(a), Control(b)])
    hs = [Constant(1.0), Constant(1.0)]
    assert taylor_test(rf3, [a, b], hs) > 1.9
