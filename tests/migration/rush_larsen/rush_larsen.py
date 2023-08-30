try:
    from dolfin import BackwardEuler
except ImportError:
    from dolfin import info_red
    info_red("Need dolfin > 1.2.0 for ode_solver test.")
    import sys; sys.exit(0)

from dolfin import *
from dolfin_adjoint import *
parameters["form_compiler"]["representation"] = "uflacs"

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
    for Solver in [RL1, RL2, GRL1, GRL2]:
        adj_reset()
        u = u0.copy(deepcopy=True, name="Solution")
        v = TestFunction(R)
        time = Constant(0.0, name="Time")
        form = lambda u, time: inner(time*u, v)*dP
        exact_u = lambda t: exp(t*t/2.0)

        dt = 0.1
        (u, xs, ys) = main(u, form(u, time), time, Solver, dt=dt)

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
            return assemble(inner(u, u)*dx)

        tlm = True
        if tlm:
            dJdm = compute_gradient_tlm(J, m, forget=False)
            minconv_tlm = taylor_test(Jhat, m, Jm, dJdm, perturbation_direction=interpolate(Constant(1.0), R), seed=1.0)
            assert minconv_tlm > 1.8

        ## Step 3. Check ADM correctness

        adm = True
        if adm:
            dJdm = compute_gradient(J, m, forget=False)
            minconv_adm = taylor_test(Jhat, m, Jm, dJdm, perturbation_direction=interpolate(Constant(1.0), R), seed=1.0)
            assert minconv_adm > 1.8
