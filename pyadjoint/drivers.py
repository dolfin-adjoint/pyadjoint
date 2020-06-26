from .enlisting import Enlist
from .tape import get_working_tape, stop_annotating


def compute_gradient(J, m, options=None, tape=None, adj_value=1.0):
    """
    Compute the gradient of J with respect to the initialisation value of m,
    that is the value of m at its creation.

    Args:
        J (AdjFloat):  The objective functional.
        m (list or instance of Control): The (list of) controls.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The derivative with respect to the control. Should be an instance of the same type as
            the control.
    """
    options = options or {}
    tape = tape or get_working_tape()
    tape.reset_variables()
    J.adj_value = adj_value
    m = Enlist(m)

    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_adj(markings=True)

    grads = [i.get_derivative(options=options) for i in m]
    return m.delist(grads)


def compute_hessian(J, m, m_dot, options=None, tape=None):
    """
    Compute the Hessian of J in a direction m_dot at the current value of m

    Args:
        J (AdjFloat):  The objective functional.
        m (list or instance of Control): The (list of) controls.
        m_dot (list or instance of the control type): The direction in which to compute the Hessian.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The second derivative with respect to the control in direction m_dot. Should be an instance of
            the same type as the control.
    """
    tape = tape or get_working_tape()
    options = options or {}

    tape.reset_tlm_values()
    tape.reset_hessian_values()

    m = Enlist(m)
    m_dot = Enlist(m_dot)
    for i, value in enumerate(m_dot):
        m[i].tlm_value = m_dot[i]

    with stop_annotating():
        tape.evaluate_tlm()

    J.block_variable.hessian_value = 0.0
    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_hessian(markings=True)

    r = [v.get_hessian(options=options) for v in m]
    return m.delist(r)


def solve_adjoint(J, tape=None, adj_value=1.0):
    """
    Solve the adjoint problem for a functional J.

    Adjoint actions are stored on each tape block as the `adj_value` attribute. In the case of
    solve blocks, intermediate adjoint solutions can also be accessed via the `adj_sol` attribute.

    Args:
        J (AdjFloat):  The objective functional.
        tape: The tape to use. Default is the current tape.
    """
    tape = tape or get_working_tape()
    tape.reset_variables()
    J.adj_value = adj_value

    with stop_annotating():
        tape.evaluate_adj(markings=False)
