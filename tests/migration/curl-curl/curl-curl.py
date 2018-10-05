from fenics import *
from fenics_adjoint import *
from distutils.version import LooseVersion

mesh = UnitCubeMesh(2, 2, 2)
import dolfin
if LooseVersion(dolfin.__version__) > LooseVersion('1.3.0'):
    dx = dx(mesh)

# Create mesh
def main(dbdt, annotate=False):

    # Define function spaces
    PN = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)
    P1 = VectorFunctionSpace(mesh, "CG", 1)

    # Define test and trial functions
    v0 = TestFunction(PN)
    u0 = TrialFunction(PN)
    v1 = TestFunction(P1)
    u1 = TrialFunction(P1)

    # Define functions
    dbdt_v = as_vector([0.0, 0.0, dbdt*dbdt])
    zero = Expression(("0.0", "0.0", "0.0"), degree=1)
    T = Function(PN)
    J = Function(P1)

    print("T: ", T.vector())
    print("J: ", J.vector())

    # Dirichlet boundary
    class DirichletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Boundary condition
    bc = DirichletBC(PN, zero, DirichletBoundary())

    # Solve eddy currents equation (using potential T)
    solve(inner(curl(v0), curl(u0))*dx == -inner(v0, dbdt_v)*dx, T, bc, annotate=annotate)

    # Solve density equation
    solve(inner(v1, u1)*dx == dot(v1, curl(T))*dx, J, annotate=annotate)

    return J

if __name__ == "__main__":
    dbdt = Constant(1.0, name="dbdt")
    J = main(dbdt, annotate=True)
    J = assemble(inner(J, J)**2*dx + inner(dbdt, dbdt)*dx)
    m = Control(dbdt)

    dJdc = compute_gradient(J, m)

    h = Constant(1.0)
    dJdc = h._ad_dot(dJdc)
    HJc = compute_hessian(J, m, h)
    HJc = h._ad_dot(HJc)

    def J(c):
        j = main(c, annotate=False)
        return assemble(inner(j, j)**2*dx + inner(c, c)*dx)

    minconv = taylor_test(J, dbdt, h, dJdc, Hm=HJc)

    assert minconv > 2.8
