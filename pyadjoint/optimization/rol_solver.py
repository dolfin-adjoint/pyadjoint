from .optimization_solver import OptimizationSolver
from ..enlisting import Enlist
from ..overloaded_type import OverloadedType
from ..tape import no_annotations

try:
    import ROL

    class ROLObjective(ROL.Objective):
        def __init__(self, rf, scale=1.):
            super(ROLObjective, self).__init__()
            self.rf = rf
            self.scale = scale

            self._val = None
            self._cache = None
            self._flag = None

        def value(self, x, tol):
            return self._val

        def gradient(self, g, x, tol):
            opts = {"riesz_representation": x.inner_product}
            self.deriv = self.rf.derivative(options=opts)
            g.dat = Enlist(self.deriv)

        def hessVec(self, hv, v, x, tol):
            opts = {"riesz_representation": x.inner_product}
            hessian_action = self.rf.hessian(v.dat, options=opts)
            hv.dat = Enlist(hessian_action)

        def update(self, x, flag, iteration):
            if hasattr(ROL, "UpdateType") and isinstance(flag, ROL.UpdateType):
                # Initial: has not been called before
                # Accept: this is the new iterate, trial has been called
                # Revert: revert to previous, trial has been called
                # Trial: candidate for next
                # Temp: temporary
                if flag in [ROL.UpdateType.Initial, ROL.UpdateType.Trial, ROL.UpdateType.Temp]:
                    self._val = self.rf(x.dat)
                    self._tape_trial = self.rf.tape.checkpoint_block_vars(self.rf.controls)
                elif flag == ROL.UpdateType.Revert:
                    # revert back to the cached value
                    self._val = self._cache
                    self.rf.tape.restore_block_vars(self._tape_cache)

                self._flag = flag

                # cache value/tape in the first instance or when accepted
                if flag in [ROL.UpdateType.Initial, ROL.UpdateType.Accept]:
                    self._cache = self._val
                    self._tape_cache = self._tape_trial

            else:
                self._val = self.rf(x.dat)

    class ROLVector(ROL.Vector):
        def __init__(self, dat, inner_product="L2"):
            super(ROLVector, self).__init__()
            self.dat = dat
            self.inner_product = inner_product

        def plus(self, yy):
            for (x, y) in zip(self.dat, yy.dat):
                x._ad_iadd(y)

        def scale(self, alpha):
            for x in self.dat:
                x._ad_imul(alpha)

        def riesz_map(self, derivs):
            dat = []
            opts = {"riesz_representation": self.inner_product}
            for deriv in Enlist(derivs):
                dat.append(deriv._ad_convert_type(deriv, options=opts))
            return dat

        def dot(self, yy):
            res = 0.
            opts = {"riesz_representation": self.inner_product}
            for (x, y) in zip(self.dat, yy.dat):
                res += x._ad_dot(y, options=opts)
            return res

        def norm(self):
            return self.dot(self) ** 0.5

        def clone(self):
            dat = []
            for x in self.dat:
                dat.append(x._ad_copy())
            res = ROLVector(dat, inner_product=self.inner_product)
            res.scale(0.0)
            return res

        def dimension(self):
            return sum(x._ad_dim() for x in self.dat)

        def reduce(self, r, r0):
            res = r0
            for x in self.dat:
                res = x._reduce(r, res)
            return res

        def applyUnary(self, f):
            for x in self.dat:
                x._applyUnary(f)

        def applyBinary(self, f, inp):
            for (x, y) in zip(self.dat, inp.dat):
                x._applyBinary(f, y)

    class ROLConstraint(ROL.Constraint):

        def __init__(self, con):
            ROL.Constraint.__init__(self)
            self.con = con

        def value(self, cvec, x, tol):
            cvec.dat = self.con.function(x.dat)

        def applyJacobian(self, jv, v, x, tol):
            self.con.jacobian_action(x.dat, v.dat[0], jv.dat)

        def applyAdjointJacobian(self, jv, v, x, tol):
            self.con.jacobian_adjoint_action(x.dat, v.dat, jv.dat[0])
            jv.dat = jv.riesz_map(jv.dat)

        def applyAdjointHessian(self, ahuv, u, v, x, tol):
            self.con.hessian_action(x.dat, u.dat[0], v.dat, ahuv.dat[0])
            ahuv.dat = ahuv.riesz_map(ahuv.dat)

    class ROLSolver(OptimizationSolver):
        """
        Use ROL to solve the given optimisation problem.
        """

        def __init__(self, problem, parameters, inner_product="L2"):
            """
            Create a new ROLSolver.

            The argument inner_product specifies the inner product to be used for
            the control space.

            """

            OptimizationSolver.__init__(self, problem, parameters)
            self.rolobjective = ROLObjective(problem.reduced_functional)
            x = [p.tape_value() for p in self.problem.reduced_functional.controls]
            self.rolvector = ROLVector(x, inner_product=inner_product)
            self.params_dict = parameters

            self.bounds = self.__get_bounds()
            self.constraints = self.__get_constraints()

        def __get_bounds(self):
            bounds = self.problem.bounds
            if bounds is None:
                return None

            controlvec = self.rolvector
            lowervec = controlvec.clone()
            uppervec = controlvec.clone()

            for i in range(len(controlvec.dat)):
                general_lb, general_ub = bounds[i]
                if isinstance(general_lb, (int, float)):
                    lowervec.dat[i]._applyUnary(lambda x: general_lb)
                else:
                    lowervec.dat[i].assign(general_lb)
                if isinstance(general_ub, (int, float)):
                    uppervec.dat[i]._applyUnary(lambda x: general_ub)
                else:
                    uppervec.dat[i].assign(general_ub)

            res = ROL.Bounds(lowervec, uppervec, 1.0)
            # FIXME: without this the lowervec and uppervec get cleaned up too
            # early.  This is a bug in PyROL and we'll hopefully figure that out
            # soon
            self.lowervec = lowervec
            self.uppervec = uppervec
            return res

        def __get_constraints(self):
            if self.problem.constraints is None:
                return ([], []), ([], [])

            eqconstraints = self.problem.constraints.equality_constraints()

            if len(eqconstraints.constraints) > 0:
                eqws = eqconstraints.output_workspace()
                if not all(isinstance(w, OverloadedType) for w in eqws):
                    raise TypeError("""To use constraints with ROL the constraint value needs
    to be an OverloadedType.""")
                eqres = [ROLConstraint(eqconstraints)], [ROLVector(eqws)]
            else:
                eqres = [], []

            ineqconstraints = self.problem.constraints.inequality_constraints()
            if len(ineqconstraints.constraints) > 0:
                ineqws = ineqconstraints.output_workspace()
                if not all(isinstance(w, OverloadedType) for w in ineqws):
                    raise TypeError("""To use constraints with ROL the constraint value needs
    to be an OverloadedType.""")
                ineqres = [ROLConstraint(ineqconstraints)], [ROLVector(ineqws)]
            else:
                ineqres = [], []

            return eqres, ineqres

        @no_annotations
        def solve(self):
            """
            Solve the optimization problem and return the optimized
            parameters.
            """

            bnd = self.bounds
            econs = self.constraints[0][0]
            emuls = self.constraints[0][1]
            icons = self.constraints[1][0]
            imuls = self.constraints[1][1]
            if len(icons) > 0:
                zeros = [i.clone() for i in imuls]
                ibnds = [ROL.Bounds(z, isLower=True) for z in zeros]
            else:
                ibnds = []

            rolproblem = ROL.OptimizationProblem(self.rolobjective,
                                                 self.rolvector,
                                                 bnd=bnd,
                                                 econs=econs,
                                                 emuls=emuls,
                                                 icons=icons,
                                                 imuls=imuls,
                                                 ibnds=ibnds)
            x = self.rolvector
            params = ROL.ParameterList(self.params_dict, "Parameters")
            self.solver = ROL.OptimizationSolver(rolproblem, params)
            self.solver.solve()
            return self.problem.reduced_functional.controls.delist(x.dat)

        def checkGradient(self):
            x = self.rolvector
            g = x.clone()
            self.rolobjective.update(x, None, None)
            self.rolobjective.gradient(g, x, 0.0)
            res = self.rolobjective.checkGradient(x, g, 7, 1)
            return res

        def getAlgorithmState(self):
            return self.solver.getAlgorithmState()

except ImportError:

    class ROLSolver(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("Could not import pyrol. Please install roltrilinos ROL using pip.")
