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

# Import cell model (rhs, init_values, default_parameters)
import tentusscher_2004_mcell as model

import random
random.seed(42)

parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

params = model.default_parameters()
state_init = model.init_values()

mesh = UnitIntervalMesh(1)
num_states = state_init.value_size()
V = VectorFunctionSpace(mesh, "CG", 1, dim=num_states)

def main(u, form, time, Scheme, dt):

    scheme = Scheme(form, u, time)
    scheme.t().assign(float(time))

    xs = [float(time)]
    ys = [u.vector().array()[0]]

    solver = PointIntegralSolver(scheme)
    solver.parameters.reset_stage_solutions = True
    solver.parameters.newton_solver.reset_each_step = True
    solver.parameters.newton_solver.maximum_iterations = 50

    for i in range(5):
        solver.step(dt)
        xs.append(float(time))
        ys.append(u.vector().array()[15])

    return (u, xs, ys)

if __name__ == "__main__":
    u0 = interpolate(state_init, V, name="InitialValue")
    Scheme = BackwardEuler

    u = Function(V, name="Solution")
    v = TestFunction(V)
    time = Constant(0.0)
    form = model.rhs(u, time, params)*dP

    ## Step 0. Check forward order-of-convergence (nothing to do with adjoints)
    check = False
    plot = False

    dt = 0.5
    u.assign(u0)
    fwd_timer = Timer("Forward run")
    (u, xs, ys) = main(u, form, time, Scheme, dt=dt)
    fwd_time = fwd_timer.stop()
    info_red("Forward time: %s" % fwd_time)


    replay = False
    if replay:
        replay_timer = Timer("Replay")
        info_blue("Checking replay correctness .. ")
        assert adjglobals.adjointer.equation_count > 0
        success = replay_dolfin(tol=1.0e-15, stop=True)
        replay_time = replay_timer.stop()
        info_red("Replay time: %s" % replay_time)
        assert success

    seed = 1.5e-4
    dtm = TimeMeasure()

    for i in [1]:
        Jform = lambda u: inner(u[i], u[i])*dx
        J = Functional(Jform(u)*dtm[FINISH_TIME])
        m = Control(u)
        Jm = assemble(Jform(u))

        def Jhat(ic):
            time = Constant(0.0)
            form = model.rhs(ic, time, params)*dP

            (u, xs, ys) = main(ic, form, time, Scheme, dt=dt)
            return assemble(Jform(u))

        adj_timer = Timer("Gradient calculation")
        dJdm = compute_gradient(J, m, forget=False)
        adj_time = adj_timer.stop()
        info_red("Gradient time: %s" % adj_time)

        minconv_adm = taylor_test(Jhat, m, Jm, dJdm, \
                                  perturbation_direction=interpolate(Constant((0.1,)*num_states), V), seed=seed)

        assert minconv_adm > 1.8

        dJdm_tlm = compute_gradient_tlm(J, m, forget=False)

        minconv_tlm = taylor_test(Jhat, m, Jm, dJdm_tlm, \
                                  perturbation_direction=interpolate(Constant((0.1,)*num_states), V), seed=seed)

        assert minconv_tlm > 1.8
