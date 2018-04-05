from __future__ import print_function
from .optimization_solver import OptimizationSolver
from . import constraints
from ..enlisting import Enlist
import numpy


try:
    import ROL
except ImportError:
    print("Could not import pyrol. Please 'pip3 install roltrilinos ROL'.")
    raise

class ROLObjective(ROL.Objective):
    def __init__(self, rf, scale=1.):
        super(ROLObjective, self).__init__()
        self.rf = rf
        self.scale = scale

    def value(self, x, tol):
        #FIXME: should check if we have evaluated here before
        return self.val

    def gradient(self, g, x, tol):
        # self.rf(x.dat)
        g.dat = g.riesz_map(self.deriv)

    def update(self, x, flag, iteration):
        self.val = self.rf(x.dat)
        self.deriv = self.rf.derivative() #forget=False, project=False)
        # pass

class ROLVector(ROL.Vector):
    def __init__(self, dat, inner_product="L2"):
        super(ROLVector, self).__init__()
        self.dat = dat
        self.inner_product = inner_product

    def plus(self, yy):
        for (x, y) in zip(self.dat, yy.dat):
            x._iadd(y)

    def scale(self, alpha):
        for x in self.dat:
            x._imul(alpha)

    def riesz_map(self, derivs):
        dat = []
        for deriv in Enlist(derivs):
            dat.append(deriv._ad_convert_type(deriv, options={"riesz_representation": self.inner_product}))
        return dat

    def dot(self, yy):
        res = 0.
        for (x, y) in zip(self.dat, yy.dat):
            res += x._ad_dot(y, options={"riesz_representation": self.inner_product})
        return res

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
        x = [p.data() for p in self.problem.reduced_functional.controls]
        self.rolvector = ROLVector(x, inner_product=inner_product)
        self.params_dict = parameters

        self.bounds = self.__get_bounds()



    def __get_bounds(self):
        bounds = self.problem.bounds
        if bounds is None:
            return None

        controlvec = self.rolvector
        lowervec = controlvec.clone()
        uppervec = controlvec.clone()

        for i in range(len(controlvec.dat)):
            x = controlvec.dat[i]
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
        # FIXME: without this the lowervec and uppervec get cleaned up too early.
        # This is a bug in PyROL and we'll hopefully figure that out soon
        self.lowervec = lowervec
        self.uppervec = uppervec
        res.test()
        return res


    def solve(self):
        """Solve the optimization problem and return the optimized parameters."""
    


        if self.problem.constraints is None:
            rolproblem = ROL.OptimizationProblem(self.rolobjective, self.rolvector,
                                                 bnd=self.bounds)
            x = self.rolvector
            params = ROL.ParameterList(self.params_dict, "Parameters")
            solver = ROL.OptimizationSolver(rolproblem, params)
            solver.solve()
            return self.problem.reduced_functional.controls.delist(x.dat)
        else:
            raise NotImplementedError()

    def checkGradient(self):
        x = self.rolvector
        g = x.clone()
        self.rolobjective.update(x, None, None)
        self.rolobjective.gradient(g, x, 0.0)
        res = self.rolobjective.checkGradient(x,g,7,1)
        return res
