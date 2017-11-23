from dolfin import *
from dolfin_adjoint import *

if __name__ == "__main__":

    mesh = UnitIntervalMesh(4)
    W = FunctionSpace(mesh, "CG", 1)

    # Create control and solution field(s)
    p = Function(W)

    Ks = Function(W)
    Ks.vector()[:] = 1.0

    ps = TrialFunction(W)
    qs = TestFunction(W)

    F = (Ks*ps + 1)*qs*dx

    # Assemble the stiffness matrix
    A = assemble(lhs(F))
    rhs = assemble(rhs(F))

    # Apply boundary conditions to the matrix
    bc = DirichletBC(W, Constant(0.0), "on_boundary")
    bc.apply(A)
    bc.apply(rhs)

    # Solve the system
    sol = PETScKrylovSolver("gmres", "amg")
    sol.set_operator(A)
    sol.solve(p.vector(), rhs)


    J = Functional(p*dx)
    m = Control(Ks)
    Jr = ReducedFunctional(J, m)

    # Check that perturbing the control changes the output
    val1 = Jr(Ks)
    Ks.vector()[:] += 1
    val2 = Jr(Ks)

    assert abs(val1 - val2) > 1e-10

    # Perform Taylor test
    Jr.taylor_test(Ks)
