from fenics import *
from fenics_adjoint import *

def test_simple_solve():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    f = Function(V)
    f.vector()[:] = 2

    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx
    L = f*v*dx

    u_ = Function(V)

    solve(a == L, u_)

    L = u_*v*dx

    u_sol = Function(V)
    solve(a == L, u_sol)

    J = assemble(u_sol**4*dx)
    Jhat = ReducedFunctional(J, f)

    tape = get_working_tape()

    h = Function(V)
    h.vector()[:] = 1
    f.set_initial_tlm_input(h)
    J.set_initial_adj_input(1.0)

    tape.evaluate()
    tape.evaluate_tlm()
    J.block_output.hessian_value = 0

    tape.evaluate_hessian()

    m = f.copy(deepcopy=True)
    dJdm = f.original_block_output.adj_value.inner(h.vector())
    Hm = f.original_block_output.hessian_value.inner(h.vector())
    assert(taylor_test(Jhat, m, h, dJdm=dJdm, Hm=Hm) > 2.9)

