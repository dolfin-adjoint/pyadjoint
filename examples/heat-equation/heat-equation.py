from __future__ import print_function
from fenics import *
from fenics_adjoint import *

mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "CG", 1)

dt = 0.001


def main(ic, annotate=False):
    u_prev = ic.copy(deepcopy=True)
    u_next = ic.copy(deepcopy=True)
    u_mid = Constant(0.5) * u_prev + Constant(0.5) * u_next

    T = 0.1
    t = 0.0

    v = TestFunction(V)

    states = [ic.copy(deepcopy=True)]
    times = [float(t)]

    timestep = 0

    while t < T:
        print("Solving for t == %s" % (t + dt))
        F = inner((u_next - u_prev) / Constant(dt), v) * dx + inner(grad(u_mid), grad(v)) * dx
        solve(F == 0, u_next, J=derivative(F, u_next), annotate=annotate)
        u_prev.assign(u_next, annotate=annotate)

        t += dt
        timestep += 1
        states.append(u_next.copy(deepcopy=True, annotate=False))
        times.append(float(t))

    return (times, states, u_prev)


if __name__ == "__main__":
    true_ic = interpolate(Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1), V)
    (times, true_states, u) = main(true_ic, annotate=False)

    guess_ic = interpolate(Expression("15 * x[0] * (1 - x[0]) * x[1] * (1 - x[1])", degree=1), V)
    (times, computed_states, u) = main(guess_ic, annotate=True)

    combined = zip(times, true_states, computed_states)

    alpha = Constant(1.0e-7)
    J = assemble(
        sum(inner(true - computed, true - computed) * dx for (time, true, computed) in combined if time >= 0.01)
        + alpha * inner(grad(guess_ic), grad(guess_ic)) * dx)

    m = Control(guess_ic)

    m_ex = Function(V, name="Temperature")
    viz = File("output/iterations.pvd")

    def derivative_cb(j, dj, m):
        m_ex.assign(m)
        viz << m_ex

    rf = ReducedFunctional(J, m)

    problem = MinimizationProblem(rf)
    parameters = {'maximum_iterations': 50}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()
