
from __future__ import print_function
__all__ = ["MoolaOptimizationProblem"]

def MoolaOptimizationProblem(rf, memoize=1):
    """Build the moola problem from the OptimizationProblem instance.
       memoize describes the number of the function and derivative
       calls to be memoized
    """

    try:
        import moola
    except ImportError:
        print("You need to install moola. Try `pip install moola`")
        raise

    class Functional(moola.Functional):
        latest_eval_hash = []
        latest_eval_eval = []
        latest_deriv_hash = []
        latest_deriv_eval = []

        def __call__(self, x):
            ''' Evaluates the functional for the given control value. '''
            if memoize > 0:
                hashx = hash(x)

                for hashp in self.latest_eval_hash:
                    if hashp == hashx:
                        moola.events.increment("Cached functional evaluation")
                        return self.latest_eval_eval[self.latest_eval_hash.index(hashp)]

                if len(self.latest_eval_hash) == memoize:
                    self.latest_eval_hash.pop(0)
                    self.latest_eval_eval.pop(0)

                self.latest_eval_hash.append(hashx)
                self.latest_eval_eval.append(rf(x.data))
                moola.events.increment("Functional evaluation")
                return self.latest_eval_eval[-1]

            else:
                moola.events.increment("Functional evaluation")
                return rf(x.data)


        def derivative(self, x):
            ''' Evaluates the gradient for the control values. '''

            if memoize > 0:
                hashx = hash(x)

                for hashp in self.latest_deriv_hash:
                    if hashp == hashx:
                        moola.events.increment("Cached derivative evaluation")
                        deriv = self.latest_deriv_eval[self.latest_deriv_hash.index(hashp)]
                        return deriv

                if len(self.latest_deriv_hash) == memoize:
                    self.latest_deriv_hash.pop(0)
                    self.latest_deriv_eval.pop(0)

                moola.events.increment("Derivative evaluation")
                D = rf.derivative(forget=False)

                if isinstance(x, moola.DolfinPrimalVector):
                    deriv = moola.DolfinDualVector(D[0], riesz_map = x.riesz_map)
                else:
                    deriv = moola.DolfinDualVectorSet([moola.DolfinDualVector(di, riesz_map = xi.riesz_map) for (di, xi) in zip(D, x.vector_list)], riesz_map = x.riesz_map)

                self.latest_deriv_hash.append(hashx)
                self.latest_deriv_eval.append(deriv)
                return deriv

            else:
                moola.events.increment("Derivative evaluation")
                D = rf.derivative(forget=False)

                if isinstance(x, moola.DolfinPrimalVector):
                    deriv = moola.DolfinDualVector(D[0], riesz_map = x.riesz_map)
                else:
                    deriv = moola.DolfinDualVectorSet([moola.DolfinDualVector(di, riesz_map = xi.riesz_map) for (di, xi) in zip(D, x.vector_list)], riesz_map = x.riesz_map)

                return deriv

        def hessian(self, x):
            ''' Evaluates the gradient for the control values. '''

            self(x)

            def moola_hessian(direction):
                assert isinstance(direction, (moola.DolfinPrimalVector,
                                              moola.DolfinPrimalVectorSet))
                moola.events.increment("Hessian evaluation")
                hes = rf.hessian(direction.data)
                return moola.DolfinDualVector(hes)

            return moola_hessian

    functional = Functional()
    return moola.Problem(functional)
