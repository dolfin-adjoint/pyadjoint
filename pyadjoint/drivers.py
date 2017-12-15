from .tape import get_working_tape, stop_annotating
from .enlisting import Enlist

def compute_gradient(J, m, block_idx=0, options=None, tape=None):
    '''Compute the gradient of J with respect to the initialisation value of m, 
    that is the value of m at its creation.'''
    options = {} if options is None else options
    tape = get_working_tape() if tape is None else tape
    tape.reset_variables()
    J.adj_value = 1.0

    with stop_annotating():
        tape.evaluate(block_idx)

    m = Enlist(m)
    grads = [i.get_derivative(options=options) for i in m]
    return m.delist(grads)


class Hessian(object):
    def __init__(self, J, m):
        self.tape = get_working_tape()
        self.functional = J
        self.controls = Enlist(m)

    def __call__(self, m_dot, options=None, tape=None):
        tape = get_working_tape() if tape is None else tape
        options = {} if options is None else options

        tape.reset_tlm_values()
        tape.reset_hessian_values()

        m_dot = Enlist(m_dot)
        for i, value in enumerate(m_dot):
            self.controls[i].tlm_value = m_dot[i]

        with stop_annotating():
            self.tape.evaluate_tlm()

        self.functional.block_variable.hessian_value = 0.0
        self.tape.evaluate_hessian()

        r = [v.get_hessian(options=options) for v in self.controls]
        return self.controls.delist(r)
