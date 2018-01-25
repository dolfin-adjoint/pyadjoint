from fenics import *
from fenics_adjoint import *


# An expression that depends on a and b
base_code = '''
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/Constant.h>
class MyCppExpression : public dolfin::Expression
{
public:
      std::shared_ptr<dolfin::Constant> a;
      std::shared_ptr<dolfin::Constant> b;
  MyCppExpression() : dolfin::Expression() {}

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const
  {
    double a_ = (double) *a;
    double b_ = (double) *b;
    values[0] = EXPRESSION;
  }
  };

  PYBIND11_MODULE(SIGNATURE, m)
    {
    py::class_<MyCppExpression, std::shared_ptr<MyCppExpression>, dolfin::Expression>
    (m, "MyCppExpression")
    .def(py::init<>())
    .def_readwrite("a", &MyCppExpression::a)
    .def_readwrite("b", &MyCppExpression::b);
}

'''

cpp_code = base_code.replace("EXPRESSION", "(x[0] - a_)*b_*b_*a_")
da_cpp_code = base_code.replace("EXPRESSION", "(x[0] - a_)*b_*b_ - b_*b_*a_")
db_cpp_code = base_code.replace("EXPRESSION", "2*(x[0] - a_)*b_*a_")
compile_cpp_code(da_cpp_code)
compile_cpp_code(cpp_code)
if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    a = Constant(0.5)
    b = Constant(0.25)

    f = CompiledExpression(compile_cpp_code(cpp_code).MyCppExpression(), a=a._cpp_object, b=b._cpp_object, degree=1)

    f.dependencies = [a, b]
    dfda = CompiledExpression(compile_cpp_code(da_cpp_code).MyCppExpression(), degree=1, )
    dfda.a = a; dfda.b = b

    dfdb = CompiledExpression(compile_cpp_code(db_cpp_code).MyCppExpression(), degree=1)
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
    assert taylor_test_multiple(rf3, [a, b], hs) > 1.9
