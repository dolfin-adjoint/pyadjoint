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
    J.block_variable.adj_value = adj_value
    m = Enlist(m)

    with stop_annotating():
        with tape.marked_control_dependents(m):
            with tape.marked_functional_dependencies(J):
                with marked_controls(m):
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
        with tape.marked_control_dependents(m):
            tape.evaluate_tlm(markings=True)

    J.block_variable.hessian_value = J.block_variable.output._ad_convert_type(
        0., options={'riesz_representation': None})

    with stop_annotating():
        with tape.marked_control_dependents(m):
            with tape.marked_functional_dependencies(J):
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


class marked_controls:
    """A context manager for marking controls.

    Note:
        This is a context manager for marking whether the class:'BlockVariable' is
        a control. On exiting the context, the class:'BlockVariable' that were
        marked as controls are automatically unmarked.

    Args:
        controls (list): A list of :class:`Control` to mark within the context manager.
    """
    def __init__(self, controls):
        self.controls = controls

    def __enter__(self):
        for control in self.controls:
            control.mark_as_control()

    def __exit__(self, *args):
        for control in self.controls:
            control.unmark_as_control()
