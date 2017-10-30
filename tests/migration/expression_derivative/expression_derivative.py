from fenics import *
from fenics_adjoint import *

pflag = False

class SourceExpression(Expression):
    def __init__(self, c, d, derivative=None, **kwargs):
        self.c = c
        self.d = d
        self.derivative = derivative
        
    def eval(self, value, x):
        if self.derivative is None:
            # Evaluate functional
            value[0] = self.c**2
            value[0] *= self.d

        elif self.derivative == self.c:
            # Evaluate derivative of functional wrt c
            value[0] = 2*self.c*self.d

        elif self.derivative == self.d:
            # Evaluate derivative of functional wrt d
            value[0] = self.c**2


if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    c = Constant(2)
    d = Constant(3)

    f = SourceExpression(c, d, degree=3)
    f._ad_ignored_attributes = ["derivative"]

    # Provide the derivative coefficients
    f.user_defined_derivatives = {c: SourceExpression(c, d, derivative=c, degree=3),
                                  d: SourceExpression(c, d, derivative=d, degree=3)}

    J = assemble(f**2*dx(domain=mesh))
    rf1 = ReducedFunctional(J, Control(c))
    rf2 = ReducedFunctional(J, Control(d))

    h = Constant(1.0)
    print("Forward: ", rf1(c))
    print("Derivative: ", rf1.derivative().values())

    assert taylor_test(rf1, c, h) > 1.9
    rf1(Constant(1.0))

    print("Forward: ", rf2(d))
    print("Derivative: ", rf2.derivative().values())

    assert taylor_test(rf2, d, h) > 1.9

