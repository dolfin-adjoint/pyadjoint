
from dolfin import *

try:
    from beatadjoint import BasicCardiacODESolver
except ImportError:
    info_red("Need beatadjoint to run")
    import sys
    sys.exit(0)

try:
    from dolfin import BackwardEuler
except ImportError:
    from dolfin import info_red
    info_red("Need dolfin > 1.2.0 for ode_solver test.")
    import sys
    sys.exit(0)

from dolfin_adjoint import *

if not hasattr(MultiStageScheme, "to_tlm"):
    info_red("Need dolfin > 1.2.0 for ode_solver test.")
    import sys
    sys.exit(0)

# Import cell model (rhs, init_values, default_parameters)
from beatadjoint.cellmodels.fitzhughnagumo import Fitzhughnagumo as model

set_log_level(ERROR)

domain = UnitIntervalMesh(1)
num_states = len(model().initial_conditions()((0.0,)))
V = None


def main(model, ics=None, annotate=False):

    params = BasicCardiacODESolver.default_parameters()
    params["theta"] = 1.0
    params["V_polynomial_family"] = "CG"
    params["S_polynomial_family"] = "CG"
    params["V_polynomial_degree"] = 1
    params["S_polynomial_degree"] = 1
    params["enable_adjoint"] = annotate
    solver = BasicCardiacODESolver(domain, None, model.num_states(), model.F,
                                   model.I, I_s=model.stimulus, params=params)

    if ics is None:
        global V
        ics = interpolate(Constant((-84.9, 0.1)), solver.VS)
        # ics = project(model.initial_conditions(), solver.VS)
        print("Initial conditions: ", ics.vector().array())
        V = solver.VS

    (vs_, vs) = solver.solution_fields()
    vs_.assign(ics)

    dt = 0.1
    T = 0.1
    solutions = solver.solve((0.0, T), dt)
    for x in solutions:
        pass

    (vs_, vs) = solver.solution_fields()
    return vs_


if __name__ == "__main__":

    u = main(model(), annotate=True)
    parameters["adjoint"]["stop_annotating"] = True

    print("Solution: ", u.vector().array())
    print("Base functional value: ", assemble(inner(u, u) * dx))

    # Step 1. Check replay correctness

    replay = False
    if replay:
        info_blue("Checking replay correctness .. ")
        assert adjglobals.adjointer.equation_count > 0
        adj_html("forward.html", "forward")
        success = replay_dolfin(tol=1.0e-15, stop=True)
        assert success

    # Step 2. Check TLM correctness

    dtm = TimeMeasure()
    J = Functional(inner(u, u) * dx * dtm[FINISH_TIME], name="norm")
    m = InitialConditionParameter("vs_")
    Jm = assemble(inner(u, u) * dx)

    def Jhat(ic):
        print("Perturbed initial condition: ", ic.vector().array())
        u = main(model(), ics=ic)
        print("Perturbed functional value: ", assemble(inner(u, u) * dx))
        print("Perturbed solution: ", u.vector().array())
        return assemble(inner(u, u) * dx)

    tlm = False
    if tlm:
        dJdm = compute_gradient_tlm(J, m, forget=False)
        set_log_level(INFO)
        minconv_tlm = taylor_test(Jhat, m, Jm, dJdm,
                                  perturbation_direction=interpolate(Constant((0.1,) * num_states), V), seed=1.0e-1)
        assert minconv_tlm > 1.8

    # Step 3. Check ADM correctness

    dJdm = compute_gradient(J, m, forget=False)
    assert dJdm is not None
    set_log_level(INFO)
    minconv_adm = taylor_test(Jhat, m, Jm, dJdm,
                              perturbation_direction=interpolate(Constant((0.1,) * num_states), V), seed=1.0e-1)
    assert minconv_adm > 1.8
