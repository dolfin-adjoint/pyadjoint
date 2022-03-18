from __future__ import print_function
import pytest
import numpy

pytest.importorskip("fenics")
pytest.importorskip("ROL")

from fenics import *
from fenics_adjoint import *


def setup_problem(n=20):
    set_log_level(LogLevel.ERROR)

    mesh = UnitSquareMesh(n, n)

    cf = MeshFunction("bool", mesh, mesh.topology().dim())
    subdomain = CompiledSubDomain(
        'std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
    subdomain.mark(cf, True)
    mesh = refine(mesh, cf)
    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 0)

    f = interpolate(Expression("x[0]+x[1]", degree=1), W)
    u = Function(V, name='State')
    v = TestFunction(V)

    F = (inner(grad(u), grad(v)) - f*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

    w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    d = 1/(2*pi**2)
    d = Expression("d*w", d=d, w=w, degree=3)

    alpha = Constant(1e-3)
    J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
    control = Control(f)

    rf = ReducedFunctional(J, control)
    params = {
        'Step': {
            'Type': 'Line Search',
        },
        'Status Test': {
            'Gradient Tolerance': 1e-11,
            'Iteration Limit': 20
        }
    }
    return rf, params, w, alpha


def test_finds_analytical_solution():
    rf, params, w, alpha = setup_problem()
    problem = MinimizationProblem(rf)
    solver = ROLSolver(problem, params, inner_product="L2")
    sol = solver.solve()
    f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w",
                            w=w, alpha=alpha, degree=3)

    assert errornorm(f_analytic, sol) < 0.02


def test_bounds_work_sensibly():
    rf, params, w, alpha = setup_problem()
    lower = 0
    upper = 0.5

    problem = MinimizationProblem(rf, bounds=(lower, upper))
    solver = ROLSolver(problem, params, inner_product="L2")
    sol1 = solver.solve().copy(deepcopy=True)
    f = rf.controls[0]
    V = f.function_space()

    lower = interpolate(Constant(lower), V)
    upper = interpolate(Constant(upper), V)
    problem = MinimizationProblem(rf, bounds=(lower, upper))
    solver = ROLSolver(problem, params, inner_product="L2")
    solver.rolvector.scale(0.0)
    sol2 = solver.solve()

    assert errornorm(sol1, sol2) < 1e-7


@pytest.mark.parametrize("contype", ["eq", "ineq"])
def test_constraint_works_sensibly(contype):
    rf, params, w, alpha = setup_problem(n=7)

    class EqVolumeConstraint(EqualityConstraint):
        """A class that enforces the volume constraint g(a) = volume - a*dx = 0."""
        def __init__(self, volume, W):
            self.volume  = float(volume)
            # The derivative of the constraint g(x) is constant (it is the diagonal of the lumped mass matrix for the control function space), so let's assemble it here once.
            # This is also useful in rapidly calculating the integral each time without re-assembling.
            self.smass  = assemble(TestFunction(W) * Constant(1) * dx)

        def function(self, m):
            integral = self.smass.inner(m[0].vector())
            return Constant(self.volume - integral)

        def jacobian_action(self, m, dm, result):
            result.assign(self.smass.inner(-dm[0].vector()))

        def jacobian_adjoint_action(self, m, dp, result):
            result[0].vector()[:] = self.smass * (-1.*dp.values()[0])

        def hessian_action(self, m, dm, dp, result):
            result[0].vector()[:] = 0.0

        def output_workspace(self):
            return Constant(0.0)

    class IneqVolumeConstraint(InequalityConstraint):
        """A class that enforces the volume constraint g(a) = volume - a*dx >= 0."""
        def __init__(self, volume, W):
            self.volume  = float(volume)
            # The derivative of the constraint g(x) is constant (it is the diagonal of the lumped mass matrix for the control function space), so let's assemble it here once.
            # This is also useful in rapidly calculating the integral each time without re-assembling.
            self.smass  = assemble(TestFunction(W) * Constant(1) * dx)

        def function(self, m):
            integral = self.smass.inner(m[0].vector())
            return Constant(self.volume - integral)

        def jacobian_action(self, m, dm, result):
            result.assign(self.smass.inner(-dm.vector()))

        def jacobian_adjoint_action(self, m, dp, result):
            result.vector()[:] = self.smass * (-1.*dp.values()[0])

        def hessian_action(self, m, dm, dp, result):
            result.vector()[:] = 0.0

        def output_workspace(self):
            return Constant(0.0)


    params = {
        'General': {
            'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
        'Step': {
            'Type': 'Augmented Lagrangian',
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            },
            'Augmented Lagrangian': {
                'Subproblem Step Type': 'Line Search',
                'Subproblem Iteration Limit': 10
            }
        },
        'Status Test': {
            'Gradient Tolerance': 1e-7,
            'Iteration Limit': 10
        }
    }

    f = rf.controls[0]
    V = f.function_space()
    vol = 0.3
    econ = EqVolumeConstraint(vol, V)
    icon = IneqVolumeConstraint(vol, V)
    bounds = (interpolate(Constant(0.0), V), interpolate(Constant(0.7), V))


    if contype == "eq":
        print("Run with equality constraint")
        problem = MinimizationProblem(rf, constraints=[econ])
    elif contype == "ineq":
        print("Run with inequality constraint")
        problem = MinimizationProblem(rf, constraints=[icon])
        return
    else:
        raise NotImplementedError

    solver = ROLSolver(problem, params, inner_product="L2")

    obj = solver.rolobjective
    x = solver.rolvector

    econ, emul = solver.constraints[0]
    icon, imul = solver.constraints[1]

    x = solver.rolvector
    x.dat[0].interpolate(Constant(0.5))
    v = x.clone()
    v.dat[0].interpolate(Constant(1.0))
    u = v.clone()
    u.plus(v)

    print("Check objective gradient and hessian")
    res0 = obj.checkGradient(x, v)
    res1 = obj.checkHessVec(x, v)
    for i in range(1, len(res0)):
        assert res0[i][3] < 0.15 * res0[i-1][3]
    assert all(r[3] < 1e-10 for r in res1)


    print("Check constraint gradient and hessian")
    if len(econ)>0:
        jv = emul[0].clone()
        jv.dat[0].assign(1.0)
        res0 = econ[0].checkApplyJacobian(x, v, jv, 5, 1)
        res1 = econ[0].checkAdjointConsistencyJacobian(jv, v, x)
        res2 = econ[0].checkApplyAdjointHessian(x, jv, u, v, 5, 1)

        assert all(r[3] < 1e-10 for r in res0)
        assert res1 < 1e-10
        assert all(r[3] < 1e-10 for r in res2)

    if len(icon)>0:
        jv = imul[0].clone()
        jv.dat[0].assign(1.0)
        res0 = icon[0].checkApplyJacobian(x, v, jv, 5, 1)
        res1 = icon[0].checkAdjointConsistencyJacobian(jv, v, x)
        res2 = icon[0].checkApplyAdjointHessian(x, jv, u, v, 5, 1)

        assert all(r[3] < 1e-10 for r in res0)
        assert res1 < 1e-10
        assert all(r[3] < 1e-10 for r in res2)

    sol1 = solver.solve().copy(deepcopy=True)
    if contype == "eq":
        assert(abs(assemble(sol1 * dx) - vol) < 1e-5)
    elif contype == "ineq":
        assert(assemble(sol1 * dx) < vol + 1e-5)
    else:
        raise NotImplementedError

@pytest.mark.parametrize("contype", ["eq", "ineq"])
def test_ufl_constraint_works_sensibly(contype):
    rf, params, w, alpha = setup_problem(n=7)

    params = {
        'General': {
            'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
        'Step': {
            'Type': 'Augmented Lagrangian',
            'Line Search': {
                'Descent Method': {
                    'Type': 'Quasi-Newton Step'
                }
            },
            'Augmented Lagrangian': {
                'Subproblem Step Type': 'Line Search',
                'Subproblem Iteration Limit': 20
            }
        },
        'Status Test': {
            'Gradient Tolerance': 1e-7,
            'Iteration Limit': 15
        }
    }

    f = rf.controls[0]
    V = f.function_space()
    vol = 0.3
    econ = UFLEqualityConstraint((vol - f.control**2)*dx, f)
    icon = UFLInequalityConstraint((vol - f.control**2)*dx, f)
    bounds = (interpolate(Constant(0.0), V), interpolate(Constant(0.7), V))


    if contype == "eq":
        print("Run with equality constraint")
        problem = MinimizationProblem(rf, constraints=[econ])
    elif contype == "ineq":
        print("Run with inequality constraint")
        problem = MinimizationProblem(rf, constraints=[icon])
        return
    else:
        raise NotImplementedError

    solver = ROLSolver(problem, params, inner_product="L2")

    econ, emul = solver.constraints[0]
    icon, imul = solver.constraints[1]

    x = solver.rolvector
    v = x.clone()
    v.dat[0].interpolate(Constant(1.0))
    u = v.clone()
    u.plus(v)


    if len(econ)>0:
        jv = emul[0].clone()
        jv.dat[0].assign(1.0)
        res0 = econ[0].checkApplyJacobian(x, v, jv, 5, 1)
        res1 = econ[0].checkAdjointConsistencyJacobian(jv, v, x)
        res2 = econ[0].checkApplyAdjointHessian(x, jv, u, v, 5, 1)

        for i in range(1, len(res0)):
            assert res0[i][3] < 0.15 * res0[i-1][3]
        assert res1 < 1e-10
        assert all(r[3] < 1e-10 for r in res2)

    if len(icon)>0:
        jv = imul[0].clone()
        jv.dat[0].assign(1.0)
        res0 = icon[0].checkApplyJacobian(x, v, jv, 5, 1)
        res1 = icon[0].checkAdjointConsistencyJacobian(jv, v, x)
        res2 = icon[0].checkApplyAdjointHessian(x, jv, u, v, 5, 1)

        for i in range(1, len(res0)):
            assert res0[i][3] < 0.15 * res0[i-1][3]
        assert res1 < 1e-10
        assert all(r[3] < 1e-10 for r in res2)

    sol1 = solver.solve().copy(deepcopy=True)
    if contype == "eq":
        assert(abs(assemble(sol1**2 * dx) - vol) < 1e-5)
    elif contype == "ineq":
        assert(assemble(sol1**2 * dx) < vol + 1e-5)
    else:
        raise NotImplementedError
