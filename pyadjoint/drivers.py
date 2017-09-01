from .tape import get_working_tape, stop_annotating
from .enlisting import Enlist

def compute_gradient(J, m, block_idx=0, options={}, tape=None):
    tape = get_working_tape() if tape is None else tape
    tape.reset_variables()
    J.set_initial_adj_input(1.0)

    with stop_annotating():
        tape.evaluate(block_idx)

    m = Enlist(m)
    grads = [i.get_derivative(options=options) for i in m]
    return m.delist(grads)


class Hessian(object):
    def __init__(self, J, m):
        self.tape = get_working_tape()
        self.functional = J
        self.control = m

    def __call__(self, m_dot, options={}):
        self.control.set_initial_tlm_input(m_dot)

        with stop_annotating():
            self.tape.evaluate_tlm()

        self.functional.block_output.hessian_value = 0
        self.tape.evaluate_hessian()

        return self.control._ad_convert_type(self.control.original_block_output.hessian_value, options)
