
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

# Import cell model (rhs, init_values, default_parameters)
import fitzhughnagumo as model

params = model.default_parameters()
state_init = model.init_values()

mesh = UnitIntervalMesh(3)
#R = FunctionSpace(mesh, "R", 0) # in my opinion, should work, but doesn't
num_states = state_init.value_size()
V = VectorFunctionSpace(mesh, "CG", 1, dim=num_states)

def main(u, form, time, scheme, dt):

    scheme.t().assign(float(time))

    xs = [float(time)]
    ys = [u.vector().array()[0]]

    solver = PointIntegralSolver(scheme)
    solver.parameters.reset_stage_solutions = True
    solver.parameters.newton_solver.reset_each_step = True

    for i in range(4):
        solver.step(dt)
        xs.append(float(time))
        ys.append(u.vector().array())

    return (u, xs, ys)

if __name__ == "__main__":

    #u = interpolate(state_init, V, name="Solution")
    #u = Function(V, name="Solution")
    u = interpolate(Constant((0.1, -84.9)), V, name="Solution")
    v = TestFunction(V)
    time = Constant(0.0)
    form = model.rhs(u, time, params)*dP

    ## Step 0. Check forward order-of-convergence (nothing to do with adjoints)
    check = False
    plot = False

    def Scheme(form, u, time):
        return RushLarsenScheme(form, u, time, order=1, generalized=True)
        return CrankNicolson(form, u, time)
        return BackwardEuler(form, u, time)

    dt = 0.01
    scheme = Scheme(form, u, time)
    (u, xs, ys) = main(u, form, time, scheme, dt=dt)

    ## Step 1. Check replay correctness

    replay = True
    if replay:
        info_blue("Checking replay correctness .. ")
        assert adjglobals.adjointer.equation_count > 0
        adj_html("forward.html", "forward")
        success = replay_dolfin(tol=1.0e-15, stop=True)
        assert success

    ## Step 2. Check TLM correctness

    dtm = TimeMeasure()
    J = Functional(inner(u, u)*dx*dtm[FINISH_TIME])
    param_idx = 0
    m = Control(params[param_idx])
    Jm = assemble(inner(u, u)*dx)


    def Jhat(a_ptb):
        ic = interpolate(Constant((0.1, -84.9)), V)
        params_ = list(params)
        params_[param_idx] = a_ptb
        time = Constant(0.0)
        form = model.rhs(ic, time, params_)*dP

        scheme = Scheme(form, ic, time)
        (u, xs, ys) = main(ic, form, time, scheme, dt=dt)
        return assemble(inner(u, u)*dx)

    dJdm = compute_gradient_tlm(J, m, forget=False)
    minconv_tlm = taylor_test(Jhat, m, Jm, dJdm, seed=1.0e1)
    assert minconv_tlm > 1.8

    ## Step 3. Check ADM correctness

    dJdm = compute_gradient(J, m, forget=False)
    print("param: ", float(params[param_idx]))
    print("dJdm: ", float(dJdm))
##    #dJdm = Constant(float(dJdm) * dt)
    ptb   = 0.0001*float(params[param_idx])
    a_ptb = Constant(float(params[param_idx]) + ptb)
    J_base  = Jhat(params[param_idx])
    J_ptb   = Jhat(a_ptb)
    dJdm_fd = (J_ptb - J_base)/ptb
    print("dJdm_fd: ", dJdm_fd)

    minconv_adm = taylor_test(Jhat, m, Jm, dJdm, seed=1.0e1)
    assert minconv_adm > 1.8
