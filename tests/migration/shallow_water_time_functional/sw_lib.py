
from fenics import *
from fenics_adjoint import *
import sys


class parameters(dict):
    '''Parameter dictionary. This subclasses dict so defaults can be set.'''

    def __init__(self, dict={}):
        self["current_time"] = 0.0
        self["theta"] = 0.5

        # Apply dict after defaults so as to overwrite the defaults
        for key, val in dict.items():
            self[key] = val

        self.required = {
            "depth": "water depth",
            "dt": "timestep",
            "finish_time": "finish time",
            "dump_period": "dump period in timesteps",
            "basename": "base name for I/O"
        }

    def check(self):
        for key, error in self.required.items():
            if key not in self:
                sys.stderr.write("Missing parameter: " + key + "\n"
                                 + "This is used to set the " + error + "\n")
                raise KeyError


def rt0(mesh):
    "Return a function space U*H on mesh from the rt0 space."

    V = FunctionSpace(mesh, 'Raviart-Thomas', 1)  # Velocity space

    H = FunctionSpace(mesh, 'DG', 0)             # Height space

    W = V * H                                        # Mixed space of both.

    return W


def p1dgp2(mesh):
    "Return a function space U*H on mesh from the rt0 space."

    V = VectorElement('DG', triangle, 1)  # Velocity space
    H = FiniteElement('CG', triangle, 2)  # Height space
    VH = MixedElement([V, H])
    W = FunctionSpace(mesh, VH)                    # Mixed space of both.

    return W


def bdfmp1dg(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDFM', 1)  # Velocity space

    H = FunctionSpace(mesh, 'DG', 1)             # Height space

    W = V * H                                        # Mixed space of both.

    return W


def bdmp0(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)  # Velocity space

    H = FunctionSpace(mesh, 'DG', 0)             # Height space

    W = V * H                                        # Mixed space of both.

    return W


def bdmp1dg(mesh):
    "Return a function space U*H on mesh from the BFDM1 space."

    V = FunctionSpace(mesh, 'BDM', 1)  # Velocity space

    H = FunctionSpace(mesh, 'DG', 1)             # Height space

    W = V * H                                        # Mixed space of both.

    return W


def construct_shallow_water(W, ds, params):
    """Construct the linear shallow water equations for the space W(=U*H) and a
    dictionary of parameters params."""
    # Sanity check for parameters.
    params.check()

    (v, q) = TestFunctions(W)
    (u, h) = TrialFunctions(W)

    n = FacetNormal(W.mesh())

    # Mass matrix
    M = inner(v, u) * dx
    M += inner(q, h) * dx

    # Divergence term.
    Ct = -inner(u, grad(q)) * dx + inner(avg(u), jump(q, n)) * dS

    # The Flather boundary condition on the left hand side
    ufl = Expression("2*eta0*sqrt(g*depth)*cos(-sqrt(g*depth)*pi/3000*t)",
                     eta0=params["eta0"], g=params["g"], depth=params["depth"], t=params["current_time"], degree=1)
    rhs_contr = inner(ufl * n, q * n) * ds(1)
    Ct += sqrt(params["g"] * params["depth"]) * inner(h, q) * ds(1)

    # The contributions of the Flather boundary condition on the right hand side
    ufr = None
    Ct += sqrt(params["g"] * params["depth"]) * inner(h, q) * ds(2)

    # Pressure gradient operator
    C = (params["g"] * params["depth"]) *\
        inner(v, grad(h)) * dx + inner(avg(v), jump(h, n)) * dS

    return (M, C + Ct, rhs_contr, ufl, ufr)


def timeloop_theta(M, G, rhs_contr, ufl, ufr, state, params, annotate=True):
    '''Solve M*dstate/dt = G*state using a theta scheme.'''

    A = M + params["theta"] * params["dt"] * G

    A_r = M - (1 - params["theta"]) * params["dt"] * G

    u_out, p_out = output_files(params["basename"])

    M_u_out, v_out, u_out_state = u_output_projector(state.function_space())

    M_p_out, q_out, p_out_state = p_output_projector(state.function_space())

    # Project the solution to P1 for visualisation.
    rhs = assemble(inner(v_out, state.split()[0]) * dx)
    solve(M_u_out, u_out_state.vector(), rhs, "cg", "sor", annotate=False)

    # Project the solution to P1 for visualisation.
    rhs = assemble(inner(q_out, state.split()[1]) * dx)
    solve(M_p_out, p_out_state.vector(), rhs, "cg", "sor", annotate=False)

    u_out << u_out_state
    p_out << p_out_state

    t = params["current_time"]
    dt = params["dt"]

    step = 0

    tmpstate = Function(state.function_space())

    j = 0
    (u_j, p_j) = split(state)
    j += 0.5 * dt * assemble(dot(u_j, u_j) * dx)

    while (t < params["finish_time"]):
        t += dt

        ufl.t = t - (1.0 - params["theta"]) * dt  # Update time for the Boundary condition expression
        step += 1
        rhs = action(A_r, state) + params["dt"] * rhs_contr

        # Solve the shallow water equations.
        solve(A == rhs, tmpstate, annotate=annotate)

        state.assign(tmpstate, annotate=annotate)

        # We solve twice, to make sure that Functional handles multiple-solves-per-timestep correctly
        solve(A == rhs, tmpstate, annotate=annotate)
        # solve(A, state.vector(), rhs, "preonly", "lu")

        state.assign(tmpstate, annotate=annotate)

        if step % params["dump_period"] == 0:

            # Project the solution to P1 for visualisation.
            rhs = assemble(inner(v_out, state.split()[0]) * dx)
            solve(M_u_out, u_out_state.vector(), rhs, "cg", "sor", annotate=False)

            # Project the solution to P1 for visualisation.
            rhs = assemble(inner(q_out, state.split()[1]) * dx)
            solve(M_p_out, p_out_state.vector(), rhs, "cg", "sor", annotate=False)

            u_out << u_out_state
            p_out << p_out_state

        if t >= params["finish_time"]:
            quad_weight = 0.5
        else:
            quad_weight = 1.0
        (u_j, p_j) = split(state)
        j += quad_weight * dt * assemble(dot(u_j, u_j) * dx)

    return j, state  # return the state and the functional's time integral contribution at the final time


def replay(state, params):

    print("Replaying forward run")

    for i in range(adjointer.equation_count):
        (fwd_var, output) = adjointer.get_forward_solution(i)

        s = libadjoint.MemoryStorage(output)
        s.set_compare(0.0)
        s.set_overwrite(True)

        adjointer.record_variable(fwd_var, s)


def adjoint(state, params, functional):

    print("Running adjoint")

    for i in range(adjointer.equation_count)[::-1]:
        print("  solving adjoint equation ", i)
        (adj_var, output) = adjointer.get_adjoint_solution(i, functional)

        s = libadjoint.MemoryStorage(output)
        adjointer.record_variable(adj_var, s)

    return output.data  # return the adjoint solution associated with the initial condition


def u_output_projector(W):
    # Projection operator for output.
    Output_V = VectorFunctionSpace(W.mesh(), 'CG', 1, dim=2)

    u_out = TrialFunction(Output_V)
    v_out = TestFunction(Output_V)

    M_out = assemble(inner(v_out, u_out) * dx)

    out_state = Function(Output_V)

    return M_out, v_out, out_state


def p_output_projector(W):
    # Projection operator for output.
    Output_V = FunctionSpace(W.mesh(), 'CG', 1)

    u_out = TrialFunction(Output_V)
    v_out = TestFunction(Output_V)

    M_out = assemble(inner(v_out, u_out) * dx)

    out_state = Function(Output_V)

    return M_out, v_out, out_state


def output_files(basename):

    # Output file
    u_out = File(basename + "_u.pvd", "compressed")
    p_out = File(basename + "_p.pvd", "compressed")

    return u_out, p_out
