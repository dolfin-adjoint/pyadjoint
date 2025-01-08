from .control import Control
from itertools import repeat


class CompositeReducedFunctional:
    def __init__(self, reduced_functional, controls_or_reducedfunctionals):
        self.rf = reduced_functional
        self.controls_or_rfs = controls_or_reducedfunctionals
        self.controls = self._denest(
            self._crf_map(repeat(0),
                          lambda rf, _: (c.copy_data() for c in rf.controls),
                          lambda c, _: c.copy_data(),))

    def __call__(self, values):
        return self.rf(
            self._crf_map(self._nest_components(values),
                          lambda rf, v: rf(v)))

    def tlm(self, m_dot):
        return self.rf.tlm(
            self._crf_map(self._nest_components(m_dot),
                          lambda rf, m: rf.tlm(m)))

    def derivative(self, adj_input=1.0):
        return self._denest_components(
            self._crf_map(self.rf.derivative(adj_input),
                          lambda rf, adj: rf.derivative(adj)))

    def hessian(self, hessian_input=0.):
        return self._denest_components(
            self._crf_map(self.rf.hessian(hessian_input),
                          lambda rf, h: rf.hessian(h)))

    def _nest_components(self, components):
        return self._crf_map(
            repeat(reversed(components)),
            lambda rf, cpts: rf.controls.delist([next(cpts) for _ in range(len(rf.controls))]),
            lambda c, cpts: next(cpts))

    def _denest_components(self, nested_components):
        return [cpt
                for cpt_list in self._crf_map(nested_components,
                                              lambda rf, cpts: cpt if rf.controls.listed else [cpt],
                                              lambda c, cpts: [cpt])
                for cpt in cpt_list]

    def _crf_map(self, values, closure_rf, closure_c=lambda c, v: v):
        return map(zip(self.controls_or_rfs, values),
                   lambda c_or_rf, v: (closure_c(c_or_rf, v) if isinstance(c_or_rf, Control)
                                       else closure_rf(c_or_rf, v)))
