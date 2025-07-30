"""Provide the abstract reduced functional, and a vanilla implementation."""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from .drivers import compute_derivative, compute_hessian, compute_tlm
from .enlisting import Enlist
from .tape import get_working_tape, stop_annotating, no_annotations
from .overloaded_type import OverloadedType, create_overloaded_object
from .adjfloat import AdjFloat
from .control import Control


class AbstractReducedFunctional(ABC):
    """Base class for reduced functionals.

    An object which encompasses computations of the form::

        Jhat(m) = J(u(m), m)

    Where `u` is the system state and `m` is a `pyadjoint.Control` or list of
    `pyadjoint.Control`, `J` is an overloaded type providing the functional value,
    and `Jhat` is a reduced functional where the explicit dependence on `u` has
    been eliminated.

    A reduced functional is callable and takes as arguments the value(s) of the
    control(s) at which it is to be evaluated.

    Despite the name, the value of a reduced functional need not be scalar. If
    the functional is non-scalar valued then derivative calculations will need
    to be seeded with an input adjoint value of the adjoint type matching the
    result of the functional.
    """

    @property
    @abstractmethod
    def controls(self) -> list[Control]:
        """Return the list of controls on which this functional depends."""

    @abstractmethod
    def __call__(
        self, values: OverloadedType | list[OverloadedType]
    ) -> OverloadedType:
        """Compute the reduced functional with supplied control value.

        Args:
            values ([OverloadedType]): If you have multiple controls this
                should be a list of new values for each control in the order
                you listed the controls to the constructor. If you have a
                single control it can either be a list or a single object.
                Each new value should have the same type as the corresponding
                control.

        Returns:
            :obj:`OverloadedType`: The computed value. Often of type
                :class:`AdjFloat`.

        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, adj_input=1.0, apply_riesz=False):
        """Return the derivative of the functional w.r.t. the control.

        Using the adjoint method, the derivative of the functional with respect
        to the control, around the last supplied value of the control, is
        computed and returned.

        Args:
            adj_input: The adjoint value to the functional result. Required if
                the functional is not scalar-valued, or if the functional is an
                intermediate result in the computation of an outer functional.
            apply_riesz: If True, apply the Riesz map of each control in order
                to return a primal gradient rather than a derivative in the
                dual space.

        Returns:
            OverloadedType: The derivative with respect to the control.
                If apply_riesz is False, should be an instance of the type dual
                to that of the control. If apply_riesz is True should have the
                same type as the control.

        """
        raise NotImplementedError

    @abstractmethod
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        """Return the action of the Hessian of the functional.

        The Hessian is evaluated w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Args:
            m_dot ([OverloadedType]): The direction in which to compute the
                action of the Hessian.
            hessian_input OverloadedType: The Hessian value for the functional result.
                Required if the functional is not scalar-valued, or if the functional
                is an intermediate result in the computation of an outer functional.
            evaluate_tlm (bool): If True, will evaluate the tangent linear model before
                evaluating the Hessian. If False, assumes that the tape is already
                populated with the tangent linear values and does not evaluate them.
            apply_riesz (bool): If True, apply the Riesz map of each control in order
                to return the (primal) Riesz representer of the Hessian
                action.

        Returns:
            OverloadedType: The action of the Hessian in the direction m_dot.
                If apply_riesz is False, should be an instance of the type dual
                to that of the control. If apply_riesz is True should have the
                same type as the control.

        """
        raise NotImplementedError

    @abstractmethod
    def tlm(self, m_dot):
        """Return the action of the tangent linear model of the functional.

        The tangent linear model is evaluated w.r.t. the control on a vector
        m_dot, around the last supplied value of the control.

        Args:
            m_dot ([OverloadedType]): The direction in which to compute the
                action of the tangent linear model.

        Returns:
            OverloadedType: The action of the tangent linear model in the direction m_dot.
                Should be an instance of the same type as the functional.
        """


def _get_extract_derivative_components(derivative_components):
    """Construct a function to pass as a pre derivative callback.

    Used when derivative components are required.
    """
    def extract_derivative_components(controls):
        controls_out = Enlist([controls[i]
                               for i in derivative_components])
        return controls_out
    return extract_derivative_components


def _get_pack_derivative_components(controls, derivative_components):
    """Construct a function to pass as a post derivative callback.

    Used when derivative components are required.
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


class ReducedFunctional(AbstractReducedFunctional):
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
        scale (float): A scaling factor applied to the functional and its
            gradient with respect to the control.
        tape (Tape): A tape object that the reduced functional will use to
            evaluate the functional and its gradients (or derivatives).
        eval_cb_pre (function): Callback function before evaluating the
            functional. Input is a list of Controls.
        eval_cb_pos (function): Callback function after evaluating the
            functional. Inputs are the functional value and a list of Controls.
        derivative_cb_pre (function): Callback function before evaluating
            derivatives. Input is a list of Controls.
            Should return a list of Controls (usually the same
            list as the input) to be passed to compute_derivative.
        derivative_cb_post (function): Callback function after evaluating
            derivatives.  Inputs are: functional.block_variable.checkpoint,
            list of functional derivatives, list of functional values.
            Should return a list of derivatives (usually the same
            list as the input) to be returned from self.derivative.
        hessian_cb_pre (function): Callback function before evaluating the Hessian.
            Input is a list of Controls.
        hessian_cb_post (function): Callback function after evaluating the Hessian.
            Inputs are the functional, a list of Hessian, and controls.
        tlm_cb_pre (function): Callback function before evaluating the tangent linear model.
            Input is a list of Controls.
        tlm_cb_post (function): Callback function after evaluating the tangent linear model.
            Inputs are the functional, the tlm result, and controls.
    """

    def __init__(self, functional, controls,
                 derivative_components=None,
                 scale=1.0, tape=None,
                 eval_cb_pre=lambda *args: None,
                 eval_cb_post=lambda *args: None,
                 derivative_cb_pre=lambda controls: controls,
                 derivative_cb_post=lambda checkpoint, derivative_components,
                 controls: derivative_components,
                 hessian_cb_pre=lambda *args: None,
                 hessian_cb_post=lambda *args: None,
                 tlm_cb_pre=lambda *args: None,
                 tlm_cb_post=lambda *args: None):
        if not isinstance(functional, OverloadedType):
            raise TypeError("Functional must be an OverloadedType.")
        self.functional = functional
        self.tape = get_working_tape() if tape is None else tape
        self._controls = Enlist(controls)
        self.derivative_components = derivative_components
        self.scale = scale
        self.eval_cb_pre = eval_cb_pre
        self.eval_cb_post = eval_cb_post
        self.derivative_cb_pre = derivative_cb_pre
        self.derivative_cb_post = derivative_cb_post
        self.hessian_cb_pre = hessian_cb_pre
        self.hessian_cb_post = hessian_cb_post
        self.tlm_cb_pre = tlm_cb_pre
        self.tlm_cb_post = tlm_cb_post

        if self.derivative_components:
            # pre callback
            self.derivative_cb_pre = _get_extract_derivative_components(
                derivative_components)
            # post callback
            self.derivative_cb_post = _get_pack_derivative_components(
                controls, derivative_components)

    @property
    def controls(self) -> list[Control]:
        return self._controls

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        values = [c.tape_value() for c in self.controls]
        controls = self.derivative_cb_pre(self.controls)

        if not controls:
            raise ValueError("""Note that the callback interface
            for derivative_cb_pre has changed. It should now return a
            list of controls (usually the same list as input.""")

        # Make sure `adj_input` is an OverloadedType
        adj_input = create_overloaded_object(adj_input)
        adj_value = adj_input._ad_mul(self.scale)

        derivatives = compute_derivative(self.functional,
                                         controls,
                                         tape=self.tape,
                                         adj_value=adj_value,
                                         apply_riesz=apply_riesz)

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
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        # Call callback
        values = [c.tape_value() for c in self.controls]
        self.hessian_cb_pre(self.controls.delist(values))

        r = compute_hessian(self.functional, self.controls, m_dot,
                            hessian_input=hessian_input, tape=self.tape,
                            evaluate_tlm=evaluate_tlm, apply_riesz=apply_riesz)

        # Call callback
        self.hessian_cb_post(self.functional.block_variable.checkpoint,
                             self.controls.delist(r),
                             self.controls.delist(values))

        return self.controls.delist(r)

    @no_annotations
    def tlm(self, m_dot):
        # Call callback
        values = [c.tape_value() for c in self.controls]
        self.tlm_cb_pre(self.controls.delist(values))

        tlm = compute_tlm(self.functional, self.controls, m_dot, tape=self.tape)

        # Call callback
        self.tlm_cb_post(self.functional.block_variable.checkpoint,
                         tlm, self.controls.delist(values))

        return tlm

    @no_annotations
    def __call__(self, values):
        values = Enlist(values)
        if len(values) != len(self.controls):
            raise ValueError(
                "values should be a list of same length as controls."
            )

        for i, value in enumerate(values):
            control_type = type(self.controls[i].control)
            if isinstance(value, (int, float)) and control_type is AdjFloat:
                value = self.controls[i].control._ad_init_object(value)
            elif not isinstance(value, control_type):
                if len(values) == 1:
                    raise TypeError(
                        "Control value must be an `OverloadedType` object with the same "
                        f"type as the control, which is {control_type}"
                    )
                else:
                    raise TypeError(
                        f"The control at index {i} must be an `OverloadedType` object "
                        f"with the same type as the control, which is {control_type}"
                    )
        # Call callback.
        self.eval_cb_pre(self.controls.delist(values))

        for i, value in enumerate(values):
            self.controls[i].update(value)

        self.tape.reset_blocks()
        blocks = self.tape.get_blocks()
        self.tape._recompute_count += 1
        with self.marked_controls():
            with stop_annotating():
                if self.tape._checkpoint_manager:
                    self.tape._checkpoint_manager.recompute(self.functional)
                else:
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

    @contextmanager
    def marked_controls(self):
        """Return a context manager which marks the active controls."""
        for control in self.controls:
            control.mark_as_control()
        try:
            yield
        finally:
            for control in self.controls:
                control.unmark_as_control()
