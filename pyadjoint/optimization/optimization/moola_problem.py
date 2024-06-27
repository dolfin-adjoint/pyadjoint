from __future__ import print_function

from ..tape import no_annotations

# TODO: There might be a better way to handle this.
try:
    import moola

    _moola_installed = True
except ImportError:
    _moola_installed = False

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

        @no_annotations
        def __call__(self, x):
            """ Evaluates the functional for the given control value. """
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

        @no_annotations
        def derivative(self, x):
            """ Evaluates the gradient for the control values. """

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
                D = rf.derivative()

                deriv = moola.convert_to_moola_dual_vector(D, x)

                self.latest_deriv_hash.append(hashx)
                self.latest_deriv_eval.append(deriv)
                return deriv

            else:
                moola.events.increment("Derivative evaluation")
                D = rf.derivative()

                deriv = moola.convert_to_moola_dual_vector(D, x)

                return deriv

        @no_annotations
        def hessian(self, x):
            """ Evaluates the gradient for the control values. """

            self(x)

            @no_annotations
            def moola_hessian(direction):
                moola.events.increment("Hessian evaluation")
                hes = rf.hessian(direction.data)
                return moola.convert_to_moola_dual_vector(hes, x)

            return moola_hessian

    functional = Functional()
    return moola.Problem(functional)


if _moola_installed:
    # Wrap all moola solve routines in no annotations
    moola.NewtonCG.solve = no_annotations(moola.NewtonCG.solve)
    moola.BFGS.solve = no_annotations(moola.BFGS.solve)
    moola.HybridCG.solve = no_annotations(moola.HybridCG.solve)
    moola.TrustRegionNewtonCG.solve = no_annotations(moola.TrustRegionNewtonCG.solve)
    moola.NonLinearCG.solve = no_annotations(moola.NonLinearCG.solve)
    moola.SteepestDescent.solve = no_annotations(moola.SteepestDescent.solve)
