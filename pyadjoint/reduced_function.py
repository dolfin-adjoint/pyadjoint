from .enlisting import Enlist
from .tape import stop_annotating, get_working_tape
from .drivers import compute_gradient


class ReducedFunction(object):
    def __init__(self, outputs, controls):
        outputs = Enlist(outputs)
        outputs = outputs.delist([output.block_variable for output in outputs])
        self.outputs = Enlist(outputs)
        self.outputs = Enlist(outputs)
        self.controls = Enlist(controls)
        self.tape = get_working_tape()

        self.adj_jac_action_cb_pre = lambda *args: None
        self.adj_jac_action_cb_post = lambda *args: None
        self.eval_cb_pre = lambda *args: None
        self.eval_cb_post = lambda *args: None

    def adj_jac_action(self, *args, options=None):
        inputs = args
        if len(inputs) != len(self.outputs):
            inputs = Enlist(inputs[0])
            if len(inputs) != len(self.outputs):
                raise TypeError("The length of inputs must match the length of function outputs.")

        values = [c.data() for c in self.controls]
        self.adj_jac_action_cb_pre(self.controls.delist(values))

        derivatives = compute_gradient(self.outputs,
                                       self.controls,
                                       options=options,
                                       tape=self.tape,
                                       adj_value=inputs)

        # Call callback
        self.adj_jac_action_cb_post(self.outputs.delist([bv.saved_output for bv in self.outputs]),
                                    self.controls.delist(derivatives),
                                    self.controls.delist(values))

        return self.controls.delist(derivatives)

    def __call__(self, *args):
        inputs = args
        if len(inputs) != len(self.controls):
            inputs = Enlist(inputs[0])
            if len(inputs) != len(self.controls):
                raise TypeError("The length of inputs must match the length of controls.")

        # Call callback.
        self.eval_cb_pre(self.controls.delist(inputs))

        for i, value in enumerate(inputs):
            self.controls[i].update(value)

        self.tape.reset_blocks()
        with self.marked_controls():
            with stop_annotating():
                self.tape.recompute()

        outputs = [output.saved_output for output in self.outputs]
        outputs = self.outputs.delist(outputs)

        # Call callback
        self.eval_cb_post(outputs, self.controls.delist(inputs))

        return outputs

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


