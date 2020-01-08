import pytest
pytest.importorskip("fenics")

from fenics import *
from fenics_adjoint import *
from pyadjoint.tape import stop_annotating
from numpy.random import rand


@pytest.mark.parametrize("parameters", [("gmres",),
                                        ("lu",),
                                        ("gmres", "ilu"),
                                        ("cg", "hypre_amg")])
def test_linear_solve(parameters):
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    f = Function(V)
    a = inner(grad(u), grad(v)) * dx
    L = f ** 2 * v * dx

    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)

    U = Function(V)
    adj = {}
    def adj_cb(adj_sol):
        adj["computed"] = adj_sol
    solve(A, U.vector(), b, *parameters, adj_cb=adj_cb)

    J = assemble(U**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    assert Jhat(f) == J

    with stop_annotating():
        adj["actual"] = Function(V)
        dFdu = A
        dJdu = assemble(derivative(U**2*dx, U))
        bc.homogenize()
        bc.apply(dJdu)
        solve(dFdu, adj["actual"].vector(), dJdu, *parameters)

    Jhat.derivative()
    assert assemble((adj["actual"] - adj["computed"])**2*dx) == 0.


@pytest.mark.parametrize("parameters", [{"solver_parameters": {"linear_solver": "lu"}},
                                        {"solver_parameters": {"linear_solver": "mumps"}},
                                        {"solver_parameters": {"linear_solver": "gmres", "preconditioner": "ilu"}},
                                        {"solver_parameters": {"linear_solver": "cg", "preconditioner": "hypre_amg"}}])
def test_var_solve(parameters):
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    f = Function(V)
    a = inner(grad(u), grad(v)) * dx
    L = f ** 2 * v * dx

    U = Function(V)
    adj = {}
    def adj_cb(adj_sol):
        adj["computed"] = adj_sol
    solve(a == L, U, bc, **parameters, adj_cb=adj_cb)

    J = assemble(U**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    assert Jhat(f) == J

    with stop_annotating():
        adj["actual"] = Function(V)
        dFdu = assemble(a)
        dJdu = assemble(derivative(U**2*dx, U))
        bc.homogenize()
        bc.apply(dFdu, dJdu)
        solve(dFdu, adj["actual"].vector(), dJdu, *tuple(parameters["solver_parameters"].values()))

    Jhat.derivative()
    assert assemble((adj["actual"] - adj["computed"])**2*dx) == 0.


@pytest.mark.parametrize("parameters", [{},
                                        {"linear_solver": "gmres",
                                         "krylov_solver": {
                                             "absolute_tolerance": 1e-3
                                         },
                                         "preconditioner": "ilu"
                                         },
                                         {"linear_solver": "cg",
                                         "krylov_solver": {
                                             "absolute_tolerance": 1e-8
                                         },
                                         "preconditioner": "hypre_amg"
                                         }])
def test_linear_variational_solver(parameters):
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    f = Function(V)
    a = inner(grad(u), grad(v)) * dx
    L = f ** 2 * v * dx

    U = Function(V)
    adj = {}
    def adj_cb(adj_sol):
        adj["computed"] = adj_sol
    problem = LinearVariationalProblem(a, L, U, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters.update(parameters)
    solver.solve(adj_cb=adj_cb)

    J = assemble(U**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    assert Jhat(f) == J

    with stop_annotating():
        adj["actual"] = Function(V)
        dFdu = assemble(a)
        dJdu = assemble(derivative(U**2*dx, U))
        bc.homogenize()
        bc.apply(dFdu, dJdu)
        solver_method = parameters.pop("linear_solver", "default")
        solver_preconditioner = parameters.pop("preconditioner", "default")
        args = (solver_method, solver_preconditioner)
        solve(dFdu, adj["actual"].vector(), dJdu, *args)

    Jhat.derivative()
    assert assemble((adj["actual"] - adj["computed"])**2*dx) == 0.


@pytest.mark.parametrize("parameters", [{},
                                        {"nonlinear_solver": "newton",
                                         "newton_solver": {
                                             "linear_solver": "mumps",
                                             "krylov_solver": {
                                                 "absolute_tolerance": 1e-3,
                                                 "maximum_iterations": 1
                                             },
                                             "preconditioner": "default"
                                         }
                                         },
                                        {"nonlinear_solver": "newton",
                                         "newton_solver": {
                                             "linear_solver": "gmres",
                                             "krylov_solver": {
                                                 "absolute_tolerance": 1e-3
                                             },
                                             "preconditioner": "ilu",
                                             "relative_tolerance": 1e-3
                                         }
                                         },
                                        {"nonlinear_solver": "snes",
                                         "snes_solver": {
                                             "linear_solver": "cg",
                                             "krylov_solver": {
                                                 "absolute_tolerance": 1e-3
                                             },
                                             "preconditioner": "ilu"
                                         }
                                         }])
def test_nonlinear_variational_solver(parameters):
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    f = Function(V)
    a = inner(grad(u), grad(v)) * dx
    L = f ** 2 * v * dx

    U = Function(V)
    adj = {}
    def adj_cb(adj_sol):
        adj["computed"] = adj_sol
    problem = NonlinearVariationalProblem(action(a, U) - L, U, bc, J=a)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters.update(parameters)
    solver.solve(adj_cb=adj_cb)

    J = assemble(U**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    assert Jhat(f) == J

    with stop_annotating():
        adj["actual"] = Function(V)
        dFdu = assemble(a)
        dJdu = assemble(derivative(U**2*dx, U))
        bc.homogenize()
        bc.apply(dFdu, dJdu)

        params = parameters
        if "newton_solver" in params:
            params = params["newton_solver"]
        elif "snes_solver" in params:
            params = params["snes_solver"]
        solver_method = params.pop("linear_solver", "default")
        solver_preconditioner = params.pop("preconditioner", "default")
        args = (solver_method, solver_preconditioner)
        solve(dFdu, adj["actual"].vector(), dJdu, *args)

    Jhat.derivative()
    assert assemble((adj["actual"] - adj["computed"])**2*dx) == 0.


def test_define_adj_solver_parameters():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    f = Function(V)
    a = inner(grad(u), grad(v)) * dx
    L = f ** 2 * v * dx

    U = Function(V)
    adj = {}
    def adj_cb(adj_sol):
        adj["computed"] = adj_sol
    problem = NonlinearVariationalProblem(action(a, U) - L, U, bc, J=a)
    solver = NonlinearVariationalSolver(problem)
    adj_args = ["cg", "hypre_amg"]
    solver.solve(adj_args=adj_args, adj_cb=adj_cb)

    J = assemble(U**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    assert Jhat(f) == J

    with stop_annotating():
        adj["actual"] = Function(V)
        dFdu = assemble(a)
        dJdu = assemble(derivative(U**2*dx, U))
        bc.homogenize()
        bc.apply(dFdu, dJdu)
        solve(dFdu, adj["actual"].vector(), dJdu, *adj_args)

    Jhat.derivative()
    assert assemble((adj["actual"] - adj["computed"])**2*dx) == 0.


def test_newton_solver_parameters():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, Constant(1), "on_boundary")

    f = Function(V)
    f.vector()[:] = rand(V.dim())
    U = Function(V)
    a = inner(grad(U), grad(v)) * dx - sin(U)*v*dx
    L = f ** 2 * v * dx

    adj = {}
    def adj_cb(adj_sol):
        adj["computed"] = adj_sol
    solver = NewtonSolver()
    F = a - L
    adj_args = ["cg", "hypre_amg"]

    class Eq(NonlinearProblem):
        def __init__(self, F, U, bc):
            super().__init__()
            self.f = F
            self.jacob = derivative(F, U, u)
            self.bc = bc

        def F(self, b, x):
            assembler = SystemAssembler(self.jacob, self.f, self.bc)
            assembler.assemble(b, x)
            #assemble(self.f, tensor=b)
            #self.bc.apply(b, x)

        def J(self, A, x):
            assembler = SystemAssembler(self.jacob, self.f, self.bc)
            assembler.assemble(A)
            #assemble(self.jacob, tensor=A)
            #self.bc.apply(A)

    problem = Eq(F, U, bc)
    solver.parameters["convergence_criterion"] = "residual"
    solver.parameters["relative_tolerance"] = 1e-6
    solver.parameters["absolute_tolerance"] = 1e-10
    solver.solve(problem, U.vector(), adj_args=adj_args, adj_cb=adj_cb)
    J = assemble(U**2*dx)

    Jhat = ReducedFunctional(J, Control(f))
    assert Jhat(f) == J

    with stop_annotating():
        adj["actual"] = Function(V)
        dFdu = assemble(adjoint(derivative(F, U)))
        dJdu = assemble(derivative(U**2*dx, U))
        bc.homogenize()
        bc.apply(dFdu, dJdu)
        solve(dFdu, adj["actual"].vector(), dJdu, *adj_args)

    Jhat.derivative()
    assert assemble((adj["actual"] - adj["computed"])**2*dx) == 0.
