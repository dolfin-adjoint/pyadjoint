from .enlisting import Enlist
from .tape import stop_annotating, get_working_tape, no_annotations
from .drivers import compute_gradient

from .overloaded_type import create_overloaded_object


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
        self.block_variables = None

    @no_annotations
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

    @no_annotations
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

        outputs = self.replay()

        # Call callback
        self.eval_cb_post(outputs, self.controls.delist(inputs))

        return outputs

    @no_annotations
    def replay(self):
        self.tape.reset_blocks()
        with self.marked_controls():
            with stop_annotating():
                self.tape.recompute()

        outputs = [create_overloaded_object(output.saved_output) for output in self.outputs]
        outputs = self.outputs.delist(outputs)

        return outputs

    def marked_controls(self):
        return marked_controls(self)

    def save_checkpoints(self):
        if self.block_variables is None:
            tape = self.tape
            bvs = set()
            for block in tape.get_blocks():
                for bv in block.get_dependencies():
                    bvs.add(bv)
                for bv in block.get_outputs():
                    bvs.add(bv)
            self.block_variables = bvs
        return [bv._checkpoint for bv in self.block_variables]

    def set_checkpoints(self, p):
        for bv, chp in zip(self.block_variables, p):
            bv._checkpoint = chp


class marked_controls(object):
    def __init__(self, rf):
        self.rf = rf

    def __enter__(self):
        for control in self.rf.controls:
            control.mark_as_control()

    def __exit__(self, *args):
        for control in self.rf.controls:
            control.unmark_as_control()


