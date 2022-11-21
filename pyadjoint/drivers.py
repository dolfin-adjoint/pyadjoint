from .enlisting import Enlist
from .tape import get_working_tape, stop_annotating


def compute_gradient(J, m, options=None, tape=None, adj_value=1.0):
    """
    Compute the gradient of J with respect to the initialisation value of m,
    that is the value of m at its creation, in the adjoint direction adj_value.

    Args:
        J (OverloadedType, list[OverloadedType]):  The objective function.
        m (Control, list[Control]): The (list of) controls.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        adj_value: The adjoint direction used to evaluate the gradient.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The derivative with respect to the control. Should be an instance of the same type as
            the control.
    """
    options = options or {}
    tape = tape or get_working_tape()
    tape.reset_variables()

    J = Enlist(J)
    m = Enlist(m)
    adj_value = Enlist(adj_value)

    for i, adj in enumerate(adj_value):
        J[i].block_variable.adj_value = adj

    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_adj(markings=True)

    grads = [v.get_derivative(options=options) for v in m]
    return m.delist(grads)


def compute_jacobian_action(J, m, m_dot, tape=None):
    """
    Compute the action of the Jacobian of J on m_dot with respect to the
    initialisation value of m, that is the value of m at its creation.

    Args:
        J (OverloadedType):  The outputs of the function.
        m (Control, list[Control]): The (list of) controls.
        tape: The tape to use. Default is the current tape.
        m_dot(OverloadedType): variation of same overloaded type as m.

    Returns:
        OverloadedType: The action on m_dot of the Jacobian of J with respect to
            the control. Should be an instance of the same type as the output of J.
    """
    tape = get_working_tape() if tape is None else tape
    tape.reset_tlm_values()

    m_dot = Enlist(m_dot)
    J = Enlist(J)
    m = Enlist(m)

    for i, tlm in enumerate(m_dot):
        m[i].tlm_value = tlm

    with stop_annotating():
        tape.evaluate_tlm()
        grads = [Ji.block_variable.tlm_value for Ji in J]

    return J.delist(grads)


def compute_hessian(J, m, m_dot, options=None, tape=None, hessian_value=0.0):
    """
    Compute the Hessian of J in a directions m_dot at the current value of m.

    Note: you must call `compute_gradient` before calling this function.

    Args:
        J (OverloadedType, list[OverloadedType]):  The objective function.
        m (Control, list[Control]): The (list of) controls.
        m_dot (list or instance of the control type): The direction in which to compute the Hessian.
        hessian_value (list or instance of the output type): Should be the
            representation for zero in the output
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
    hessian_value = Enlist(hessian_value)
    J = Enlist(J)

    for i, value in enumerate(m_dot):
        m[i].tlm_value = m_dot[i]

    with stop_annotating():
        tape.evaluate_tlm()

    for i in range(len(hessian_value)):
        J[i].block_variable.hessian_value = hessian_value[i]

    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_hessian(markings=True)

    r = [v.get_hessian(options=options) for v in m]
    return m.delist(r)


def solve_adjoint(J, tape=None, adj_value=1.0):
    """
    Solve the adjoint problem for a functional J.

    This traverses the entire tape backwards, unlike `compute_gradient` which only works out those
    parts of the adjoint necessary to compute the sensitivity with respect to the specified control.
    As a result sensitivities with respect to all intermediate states are accumulated in the
    `adj_value` attribute of the associated block-variables. The adjoint solution of each solution
    step is stored in the `adj_sol` attribute of the corresponding solve block.

    Args:
        J (AdjFloat):  The objective functional.
        tape: The tape to use. Default is the current tape.
    """
    tape = tape or get_working_tape()
    tape.reset_variables()
    J.adj_value = adj_value

    with stop_annotating():
        tape.evaluate_adj(markings=False)
