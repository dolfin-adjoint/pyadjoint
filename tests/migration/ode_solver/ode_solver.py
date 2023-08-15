
try:
    from dolfin import BackwardEuler
except ImportError:
    from dolfin import info_red
    info_red("Need dolfin > 1.2.0 for ode_solver test.")
    import sys; sys.exit(0)

from dolfin import *
from dolfin_adjoint import *

if not hasattr(MultiStageScheme, "to_tlm"):
    info_red("Need dolfin > 1.2.0 for ode_solver test.")
    import sys; sys.exit(0)

mesh = UnitIntervalMesh(1)
#R = FunctionSpace(mesh, "R", 0) # in my opinion, should work, but doesn't
R = FunctionSpace(mesh, "CG", 1)

def main(u, form, time, Solver, dt):

    scheme = Solver(form, u, time)
    scheme.t().assign(float(time))

    xs = [float(time)]
    ys = [u.vector().array()[0]]

    solver = PointIntegralSolver(scheme)
    solver.parameters.reset_stage_solutions = True
    solver.parameters.newton_solver.reset_each_step = True

    for i in range(int(0.2/dt)):
        solver.step(dt)
        xs.append(float(time))
        ys.append(u.vector().array()[0])

    return (u, xs, ys)

if __name__ == "__main__":
    u0 = interpolate(Constant(1.0), R, name="InitialValue")
    c_f = 1.0
    c  = interpolate(Constant(1.0), R, name="GrowthRate")
    Solver = RK4

    u = u0.copy(deepcopy=True, name="Solution")
    v = TestFunction(R)
    time = Constant(0.0)
    # FIXME: make this work in the forward code:
    #expr = Expression("t", t=time)
    #form = inner(expr(u, v)*dP
    form = lambda u, time: inner(time*u, v)*dP
    exact_u = lambda t: exp(t*t/2.0)
    #form = lambda u, time: inner(u, v)*dP
    #exact_u = lambda t: exp(t)

    ## Step 0. Check forward order-of-convergence (nothing to do with adjoints)
    check = False
    plot = False

    if check:
        if plot:
            import matplotlib.pyplot as plt

        dts = [0.1, 0.05, 0.025]

        errors = []
        for dt in dts:
            u.assign(u0)
            time.assign(0.0)
            adj_reset()
            (u, xs, ys) = main(u, form(u, time), time, Solver, dt=dt)

            exact_ys = [exact_u(t) for t in xs]
            errors.append(abs(ys[-1] - exact_ys[-1]))

            if plot:
                plt.plot(xs, ys, label="Approximate solution (dt %s)" % dt)
                if dt == dts[-1]:
                    plt.plot(xs, exact_ys, label="Exact solution")

        print("Errors: ", errors)
        print("Convergence order: ", convergence_order(errors))

        assert min(convergence_order(errors)) > 0.8

        if plot:
            plt.legend(loc="best")
            plt.show()
    else:
        dt = 0.1
        (u, xs, ys) = main(u, form(u, time), time, Solver, dt=dt)
        print("Solution: ", ys[-1])

    ## Step 1. Check replay correctness

    replay = True
    if replay:
        assert adjglobals.adjointer.equation_count > 0
        adj_html("forward.html", "forward")
        success = replay_dolfin(tol=1.0e-15, stop=True)
        assert success

    ## Step 2. Check TLM correctness

    dtm = TimeMeasure()
    J = Functional(inner(u, u)*dx*dtm[FINISH_TIME])
    m = Control(u)
    assert m.tape_value().vector()[0] == u0.vector()[0]
    Jm = assemble(inner(u, u)*dx)

    def Jhat(ic):
        time = Constant(0.0)
        (u, xs, ys) = main(ic, form(ic, time), time, Solver, dt=dt)
        print("Perturbed functional value: ", assemble(inner(u, u)*dx))
        return assemble(inner(u, u)*dx)

    dJdm = compute_gradient_tlm(J, m, forget=False)
    minconv_tlm = taylor_test(Jhat, m, Jm, dJdm, perturbation_direction=interpolate(Constant(1.0), R), seed=1.0)
    assert minconv_tlm > 1.8

    ## Step 3. Check ADM correctness

    dJdm = compute_gradient(J, m, forget=False)
    minconv_adm = taylor_test(Jhat, m, Jm, dJdm, perturbation_direction=interpolate(Constant(1.0), R), seed=1.0)
    assert minconv_adm > 1.8
