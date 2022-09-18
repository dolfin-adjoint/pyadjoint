from functools import wraps
from .drivers import compute_gradient, compute_hessian
from .enlisting import Enlist
from .tape import get_working_tape, stop_annotating, no_annotations
from .overloaded_type import OverloadedType


class ReducedFunction(object):
    """Class representing the reduced function.

    A reduced function maps a control value to the provided function.
    It may also be used to compute the derivative of the function with
    respect to the control.

    Args:
        outputs (:obj:`OverloadedType`): An instance of an OverloadedType,
            usually :class:`AdjFloat`. This should be the return value of the
            function you want to reduce. It may also be a list instead of a
            single OverloadedType instance.
        controls (list[Control]): A list of Control instances, which you want
            to map to the outputs. It is also possible to supply a single Control
            instance instead of a list.
    """
    def __init__(
        self,
        outputs,
        controls,
        tape=None,
        eval_cb_pre=None,
        eval_cb_post=None,
        jac_action_cb_pre=None,
        jac_action_cb_post=None,
        adj_jac_action_cb_pre=None,
        adj_jac_action_cb_post=None,
        hess_action_cb_pre=None,
        hess_action_cb_post=None,
    ):
        self.output_vals = Enlist(outputs)
        for out in self.output_vals:
            if not isinstance(out, OverloadedType):
                raise TypeError(f"output must be an OverloadedType: {out}")
        self.outputs = Enlist([out.block_variable for out in self.output_vals])
        self.outputs.listed = self.output_vals.listed

        self.controls = Enlist(controls)
        self.tape = get_working_tape() if tape is None else tape

        nothing = lambda *args: None
        self.eval_cb_pre = nothing if eval_cb_pre is None else eval_cb_pre
        self.eval_cb_post = nothing if eval_cb_post is None else eval_cb_post
        self.jac_action_cb_pre = (
            nothing if jac_action_cb_pre is None else jac_action_cb_pre
        )
        self.jac_action_cb_post = (
            nothing if jac_action_cb_post is None else jac_action_cb_post
        )
        self.adj_jac_action_cb_pre = (
            nothing if adj_jac_action_cb_pre is None else adj_jac_action_cb_pre
        )
        self.adj_jac_action_cb_post = (
            nothing if adj_jac_action_cb_post is None else adj_jac_action_cb_post
        )
        self.hess_action_cb_pre = (
            nothing if hess_action_cb_pre is None else hess_action_cb_pre
        )
        self.hess_action_cb_post = (
            nothing if hess_action_cb_post is None else hess_action_cb_post
        )

    @no_annotations
    def adj_jac_action(self, adj_input, options=None):
        """Returns the action of the adjoint Jacobian of the function w.r.t. the
        control on a vector adj_input.

        Using the adjoint method, the action of the adjoint Jacobian with
        respect to the control, around the last supplied value of the control,
        is computed and returned.

        Args:
            options (dict): A dictionary of options. To find a list of available
                options have a look at the specific control type.

        Returns:
            OverloadedType: The action of the Jacobian with respect to the control.
                Should be an instance of the same type as the control.
        """
        adj_input = Enlist(adj_input)
        if len(adj_input) != len(self.outputs):
            raise ValueError(
                "The length of adj_input must match the length of function outputs."
            )

        values = [c.tape_value() for c in self.controls]
        self.adj_jac_action_cb_pre(self.controls.delist(values))

        derivatives = compute_gradient(
            self.output_vals,
            self.controls,
            options=options,
            tape=self.tape,
            adj_value=adj_input,
        )

        # Call callback
        self.adj_jac_action_cb_post(
            self.outputs.delist([bv.saved_output for bv in self.outputs]),
            self.controls.delist(derivatives),
            self.controls.delist(values),
        )

        return self.controls.delist(derivatives)

    @no_annotations
    def hess_action(self, m_dot, adj_input, options=None):
        """Returns the action of the Hessian of the function w.r.t. the
        control on the vectors m_dot and adj_input.

        Using the second-order adjoint method, the action of the Hessian of the
        function with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Args:
            m_dot ([OverloadedType]): The direction in which to compute the
                action of the Hessian.
            adj_input ([OverloadedType]): The adjoint direction in which to
                compute the action of the Hessian.
            options (dict): A dictionary of options. To find a list of
                available options have a look at the specific control type.

        Returns:
            OverloadedType: The action of the Hessian in the direction m_dot.
                Should be an instance of the same type as the control.
        """
        m_dot = Enlist(m_dot)
        if len(m_dot) != len(self.controls):
            raise ValueError(
                "The length of m_dot must match the length of function controls."
            )

        adj_input = Enlist(adj_input)
        if len(adj_input) != len(self.outputs):
            raise ValueError(
                "The length of adj_input must match the length of function outputs."
            )

        values = [c.tape_value() for c in self.controls]
        self.hess_action_cb_pre(self.controls.delist(values))

        compute_gradient(
            self.output_vals,
            self.controls,
            options=options,
            tape=self.tape,
            adj_value=adj_input,
        )

        # TODO: there should be a better way of generating hessian_value.
        zero = [0 * v for v in adj_input]
        hessian = compute_hessian(
            self.output_vals,
            self.controls,
            m_dot,
            options=options,
            tape=self.tape,
            hessian_value=zero,
        )

        # Call callback
        self.hess_action_cb_post(
            self.outputs.delist([bv.saved_output for bv in self.outputs]),
            self.controls.delist(hessian),
            self.controls.delist(values),
        )

        return self.controls.delist(hessian)

    @no_annotations
    def __call__(self, inputs):
        """Computes the reduced function with supplied control value.

        Args:
            inputs ([OverloadedType]): If you have multiple controls this should be a list of
                new values for each control in the order you listed the controls to the constructor.
                If you have a single control it can either be a list or a single object.
                Each new value should have the same type as the corresponding control.

        Returns:
            :obj:`OverloadedType`: The computed value. Typically of instance
                of :class:`AdjFloat`.

        """
        inputs = Enlist(inputs)
        if len(inputs) != len(self.controls):
            raise ValueError("The length of inputs must match the length of controls.")

        # Call callback.
        self.eval_cb_pre(self.controls.delist(inputs))

        for i, value in enumerate(inputs):
            self.controls[i].update(value)

        self.tape.reset_blocks()
        with self.marked_controls():
            with stop_annotating():
                self.tape.recompute()

        output_vals = self.outputs.delist(
            [output.saved_output for output in self.outputs]
        )

        # Call callback
        self.eval_cb_post(output_vals, self.controls.delist(inputs))

        return output_vals

    def optimize_tape(self):
        self.tape.optimize(
            controls=self.controls,
            functionals=self.output_vals
        )

    def marked_controls(self):
        return marked_controls(self)


class ReducedFunctional(ReducedFunction):
    """Class representing the reduced functional.

    A reduced functional maps a control value to the provided functional.
    It may also be used to compute the derivative of the functional with
    respect to the control.

    Args:
        functional (:obj:`OverloadedType`): An instance of an OverloadedType,
            usually :class:`AdjFloat`. This should be the return value of the
            functional you want to reduce.
        controls (list[Control]): A list of Control instances, which you want
            to map to the functional. It is also possible to supply a single Control
            instance instead of a list.
    """

    def __init__(self, functional, controls, scale=1.0,
                 derivative_cb_pre=None,
                 derivative_cb_post=None,
                 hessian_cb_pre=None,
                 hessian_cb_post=None,
                 **kwargs):
        self.functional = functional
        self.scale = scale
        return super().__init__(
            functional,
            controls,
            adj_jac_action_cb_pre=derivative_cb_pre,
            adj_jac_action_cb_post=derivative_cb_post,
            hess_action_cb_pre=hessian_cb_pre,
            hess_action_cb_post=hessian_cb_post,
            **kwargs
        )

    @property
    def eval_cb_post(self):
        return self._eval_cb_post

    @eval_cb_post.setter
    def eval_cb_post(self, func):
        @wraps(func)
        def eval_cb_post_with_scale(func_value, *args):
            return func(func_value * self.scale, *args)
        self._eval_cb_post = eval_cb_post_with_scale

    @property
    def derivative_cb_pre(self):
        return self.adj_jac_action_cb_pre

    @derivative_cb_pre.setter
    def derivative_cb_pre(self, val):
        self.adj_jac_action_cb_pre = val

    @property
    def derivative_cb_post(self):
        return self.adj_jac_action_cb_post

    @derivative_cb_post.setter
    def derivative_cb_post(self, val):
        self.adj_jac_action_cb_post = val

    @property
    def hessian_cb_pre(self):
        return self.hess_action_cb_pre

    @hessian_cb_pre.setter
    def hessian_cb_pre(self, val):
        self.hess_action_cb_pre = val

    @property
    def hessian_cb_post(self):
        return self.hess_action_cb_post

    @hessian_cb_post.setter
    def hessian_cb_post(self, val):
        self.hess_action_cb_post = val

    def derivative(self, options={}):
        """Returns the derivative of the functional w.r.t. the control.

        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the control,
        is computed and returned.

        Args:
            options (dict): A dictionary of options. To find a list of available options
                have a look at the specific control type.

        Returns:
            OverloadedType: The derivative with respect to the control.
                Should be an instance of the same type as the control.

        """
        return self.adj_jac_action(self.scale)

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
        return self.hess_action(m_dot, self.scale)

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
        val = super().__call__(values)
        return val * self.scale


class marked_controls(object):
    def __init__(self, rf):
        self.rf = rf

    def __enter__(self):
        for control in self.rf.controls:
            control.mark_as_control()

    def __exit__(self, *args):
        for control in self.rf.controls:
            control.unmark_as_control()
