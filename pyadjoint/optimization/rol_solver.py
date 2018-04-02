from __future__ import print_function
from .optimization_solver import OptimizationSolver
from . import constraints
from ..enlisting import Enlist
import numpy

from fenics import * 
# FIXME: This should be replaced by "from backend import *",
# but that doesn't seem to be a thing in pyadjoint anymore.


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
        self.dims = []
        for x in self.dat:
            if isinstance(x, Function):
                self.dims.append(x.vector().local_size())
            elif isinstance(x, Constant):
                self.dims.append(x.value_size())
            elif isinstance(x, numpy.ndarray):
                self.dims.append(len(x))
            else:
                raise NotImplementedError

    def plus(self, yy):
        for (x, y) in zip(self.dat, yy.dat):
            if isinstance(x, Function):
                assert isinstance(y, Function)
                xvec = x.vector()
                yvec = y.vector()
                xvec += yvec
            elif isinstance(x, Constant):
                x += y
            elif isinstance(x, numpy.ndarray):
                assert isinstance(y, numpy.ndarray)
                x += y
            else:
                raise NotImplementedError

    def scale(self, alpha):
        for x in self.dat:
            if isinstance(x, Function):
                xvec = x.vector()
                xvec *= alpha
            elif isinstance(x, Constant) or isinstance(x, numpy.ndarray):
                x *= alpha
            else:
                raise NotImplementedError

    def riesz_map(self, derivs):
        dat = []
        for deriv in Enlist(derivs):
            if isinstance(deriv, Function) and self.inner_product!="l2":
                V = deriv.function_space()
                u = TrialFunction(V)
                v = TestFunction(V)
                if self.inner_product=="L2":
                    M = assemble(inner(u, v)*dx)
                elif self.inner_product=="H1":
                    M = assemble((inner(u, v) + inner(grad(u), grad(v)))*dx)
                else:
                    raise ValueError("Unknown inner product %s".format(inner_product))
                proj = Function(V)
                solve(M, proj.vector(), deriv.vector())
                dat.append(proj)
            else:
                dat.append(deriv)
        return dat

    def dot(self, yy):
        res = 0.
        for (x, y) in zip(self.dat, yy.dat):
            if isinstance(x, Function):
                assert isinstance(y, Function)
                if self.inner_product == "H1":
                    res += assemble((inner(x, y) + inner(grad(x), grad(y)))*dx)
                elif self.inner_product == "L2":
                    res += assemble(inner(x, y)*dx)
                elif self.inner_product == "l2":
                    res += x.vector().inner(y.vector())
                else:
                    raise ValueError("No inner product specified for DolfinVectorSpace")

            elif isinstance(x, Constant):
                res += float(x)*float(y)
            elif isinstance(x, numpy.ndarray):
                assert isinstance(y, numpy.ndarray)
                res += numpy.inner(x, y)
            else:
                raise NotImplementedError
        return res

    def clone(self):
        dat = []
        for x in self.dat:
            if isinstance(x, Function):
                x.vector().apply("")
                dat.append(Function.copy(x, deepcopy=True))
            elif isinstance(x, Constant):
                dat.append(Constant(float(x)))
            elif isinstance(x, numpy.ndarray):
                dat.append(numpy.array(x))
            else:
                raise NotImplementedError
        res = ROLVector(dat, inner_product=self.inner_product)
        res.scale(0.0)
        return res

    def dimension(self):
        return sum(self.dims)

    def basis(self, i):
        raise NotImplementedError()

    def reduce(self, r, r0):
        res = r0
        for i in range(len(self.dat)):
            x = self.dat[i]
            if isinstance(x, Function):
                vecx = x.vector()
                tempx = vecx.get_local()
                for i in range(len(tempx)):
                    res = r(tempx[i], res)
                vecx.set_local(tempx)
                # # FIXME: is this really needed?
                # vecx.apply("insert")
                # x.vector().apply("")
            elif isinstance(x, Constant):
                tempx = x.values()
                for i in range(len(tempx)):
                    res = r(tempx[i], res)
                x.assign(tempx)
            elif isinstance(x, numpy.ndarray):
                for i in range(len(x)):
                    res = r(x[i], res)
            else:
                raise NotImplementedError
        return res


    def applyUnary(self, f):
        for i in range(len(self.dat)):
            x = self.dat[i]
            if isinstance(x, Function):
                vecx = x.vector()
                tempx = vecx.get_local()
                for i in range(len(tempx)):
                    tempx[i] = f(tempx[i])
                vecx.set_local(tempx)
                # # FIXME: is this really needed?
                # vecx.apply("insert")
                # x.vector().apply("")
            elif isinstance(x, Constant):
                tempx = x.values()
                for i in range(len(tempx)):
                    tempx[i] = f(tempx[i])
                x.assign(tempx)
            elif isinstance(x, numpy.ndarray):
                for i in range(len(x)):
                    x[i] = f(x[i])
            else:
                raise NotImplementedError

    def applyBinary(self, f, inp):
        for i in range(len(self.dat)):
            x = self.dat[i]
            y = inp.dat[i]
            if isinstance(x, Function):
                vecx = x.vector()
                vecy = y.vector()
                tempx = vecx.get_local()
                tempy = vecy.get_local()
                for i in range(len(tempx)):
                    tempx[i] = f(tempx[i], tempy[i])
                vecx.set_local(tempx)
                # # FIXME: is this really needed?
                # vecx.apply("insert")
                # x.vector().apply("")
            elif isinstance(x, Constant):
                tempx = x.values()
                tempy = y.values()
                for i in range(len(tempx)):
                    tempx[i] = f(tempx[i], tempy[i])
                x.assign(tempx)
            elif isinstance(x, numpy.ndarray):
                for i in range(len(x)):
                    x[i] = f(x[i], y[i])
            else:
                raise NotImplementedError



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
            general_lb, general_ub = bounds[i] # could be float, Constant, or Function
            if isinstance(x, (Function, Constant)):
                if isinstance(general_lb, (int, float)):
                    lowervec.dat[i].assign(Constant(general_lb))
                else:
                    lowervec.dat[i].assign(general_lb)
                if isinstance(general_ub, (int, float)):
                    uppervec.dat[i].assign(Constant(general_ub))
                else:
                    uppervec.dat[i].assign(general_ub)
            elif isinstance(x, (float, int)):
                lowervec.dat[i] = float(general_lb)
                uppervec.dat[i] = float(general_ub)
            else:
                raise TypeError("Unknown bound type %s" % general_lb.__class__)
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
            return x.dat
        else:
            raise NotImplementedError()

    def checkGradient(self):
        x = self.rolvector
        g = x.clone()
        self.rolobjective.update(x, None, None)
        self.rolobjective.gradient(g, x, 0.0)
        res = self.rolobjective.checkGradient(x,g,7,1)
        return res
