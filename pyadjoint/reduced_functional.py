from .drivers import compute_gradient, compute_hessian
from .enlisting import Enlist
from .tape import get_working_tape, stop_annotating, no_annotations
from .overloaded_type import OverloadedType, create_overloaded_object


def _get_extract_derivative_components(derivative_components):
    """
    Construct a function to pass as a pre derivative callback
    when derivative components are required.
    """
    def extract_derivative_components(controls):
        controls_out = Enlist([controls[i]
                               for i in derivative_components])
        return controls_out
    return extract_derivative_components


def _get_pack_derivative_components(controls, derivative_components):
    """
    Construct a function to pass as a post derivative callback
    when derivative components are required.
    """
    def pack_derivative_components(checkpoint, derivatives, values):
        derivatives_out = []
        count = 0
        for i, control in enumerate(controls):
            if i in derivative_components:
                derivatives_out.append(derivatives[count])
                count += 1
            else:
                zero_derivative = control._ad_copy()
                zero_derivative *= 0.
                derivatives_out.append(zero_derivative)
        return derivatives_out
    return pack_derivative_components


class ReducedFunctional(object):
    """Class representing the reduced functional.

    A reduced functional maps a control value to the provided functional.
    It may also be used to compute the derivative of the functional with
    respect to the control.

    Args:
        functional (:obj:`OverloadedType`): An instance of an OverloadedType,
            usually :class:`AdjFloat`. This should be the return value of the
            functional you want to reduce.
        controls (list[Control]): A list of Control instances, which you want
            to map to the functional. It is also possible to supply a single
            Control instance instead of a list.
        derivative_components (tuple of int): The indices of the controls with
            respect to which to take the derivative. By default, the derivative
            is taken with respect to all controls. If present, it overwrites
            derivative_cb_pre and derivative_cb_post.
        derivative_cb_pre (function): Callback function before evaluating
            derivatives. Input is a list of Controls.
            Should return a list of Controls (usually the same
            list as the input) to be passed to compute_gradient.
        derivative_cb_post (function): Callback function after evaluating
            derivatives.  Inputs are: functional.block_variable.checkpoint,
            list of functional derivatives, list of functional values.
            Should return a list of derivatives (usually the same
            list as the input) to be returned from self.derivative.
    """

    def __init__(self, functional, controls,
                 derivative_components=None,
                 scale=1.0, tape=None,
                 eval_cb_pre=lambda *args: None,
                 eval_cb_post=lambda *args: None,
                 derivative_cb_pre=lambda controls: controls,
                 derivative_cb_post=lambda checkpoint, derivative_components, controls: derivative_components,
                 hessian_cb_pre=lambda *args: None,
                 hessian_cb_post=lambda *args: None):
        if not isinstance(functional, OverloadedType):
            raise TypeError("Functional must be an OverloadedType.")
        self.functional = functional
        self.tape = get_working_tape() if tape is None else tape
        self.controls = Enlist(controls)
        self.derivative_components = derivative_components
        self.scale = scale
        self.eval_cb_pre = eval_cb_pre
        self.eval_cb_post = eval_cb_post
        self.derivative_cb_pre = derivative_cb_pre
        self.derivative_cb_post = derivative_cb_post
        self.hessian_cb_pre = hessian_cb_pre
        self.hessian_cb_post = hessian_cb_post

        if self.derivative_components:
            # pre callback
            self.derivative_cb_pre = _get_extract_derivative_components(
                derivative_components)
            # post callback
            self.derivative_cb_post = _get_pack_derivative_components(
                controls, derivative_components)

    def derivative(self, adj_input=1.0, options={}):
        """Returns the derivative of the functional w.r.t. the control.
        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the
        control, is computed and returned.

        Args:
            options (dict): A dictionary of options. To find a list of
                available options have a look at the specific control type.

        Returns:
            OverloadedType: The derivative with respect to the control.
                Should be an instance of the same type as the control.

        """

        # Call callback
        values = [c.tape_value() for c in self.controls]
        controls = self.derivative_cb_pre(self.controls)

        if not controls:
            raise ValueError("""Note that the callback interface
            for derivative_cb_pre has changed. It should now return a
            list of controls (usually the same list as input.""")

        # Scale adjoint input
        with stop_annotating():
            # Make sure `adj_input` is an OverloadedType
            adj_input = create_overloaded_object(adj_input)
            adj_value = adj_input._ad_mul(self.scale)

        derivatives = compute_gradient(self.functional,
                                       controls,
                                       options=options,
                                       tape=self.tape,
                                       adj_value=adj_value)

        # Call callback
        derivatives = self.derivative_cb_post(
            self.functional.block_variable.checkpoint,
            derivatives,
            values)

        if not derivatives:
            raise ValueError("""Note that the callback interface
            for derivative_cb_post has changed. It should now return a
            list of derivatives, usually the same list as input.""")

        return self.controls.delist(derivatives)

    @no_annotations
    def hessian(self, m_dot, options={}):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Args:
            m_dot ([OverloadedType]): The direction in which to compute the
                action of the Hessian.
            options (dict): A dictionary of options. To find a list of
                available options have a look at the specific control type.

        Returns:
            OverloadedType: The action of the Hessian in the direction m_dot.
                Should be an instance of the same type as the control.
        """
        # Call callback
        values = [c.tape_value() for c in self.controls]
        self.hessian_cb_pre(self.controls.delist(values))

        r = compute_hessian(self.functional, self.controls, m_dot, options=options, tape=self.tape)

        # Call callback
        self.hessian_cb_post(self.functional.block_variable.checkpoint,
                             self.controls.delist(r),
                             self.controls.delist(values))

        return self.controls.delist(r)

    @no_annotations
    def __call__(self, values):
        """Computes the reduced functional with supplied control value.

        Args:
            values ([OverloadedType]): If you have multiple controls this should be a list of
                new values for each control in the order you listed the controls to the constructor.
                If you have a single control it can either be a list or a single object.
                Each new value should have the same type as the corresponding control.

        Returns:
            :obj:`OverloadedType`: The computed value. Typically of instance
                of :class:`AdjFloat`.

        """
        values = Enlist(values)
        if len(values) != len(self.controls):
            raise ValueError("values should be a list of same length as controls.")

        # Call callback.
        self.eval_cb_pre(self.controls.delist(values))

        for i, value in enumerate(values):
            self.controls[i].update(value)

        self.tape.reset_blocks()
        blocks = self.tape.get_blocks()
        with self.marked_controls():
            with stop_annotating():
                for i in self.tape._bar("Evaluating functional").iter(
                    range(len(blocks))
                ):
                    blocks[i].recompute()

        # ReducedFunctional can result in a scalar or an assembled 1-form
        func_value = self.functional.block_variable.saved_output
        # Scale the underlying functional value
        func_value *= self.scale

        # Call callback
        self.eval_cb_post(func_value, self.controls.delist(values))

        return func_value

    def optimize_tape(self):
        self.tape.optimize(
            controls=self.controls,
            functionals=[self.functional]
        )

    def marked_controls(self):
        return marked_controls(self)


class marked_controls(object):
    def __init__(self, rf):
        self.rf = rf

    def __enter__(self):
        for control in self.rf.controls:
            control.mark_as_control()

    def __exit__(self, *args):
        for control in self.rf.controls:
            control.unmark_as_control()
