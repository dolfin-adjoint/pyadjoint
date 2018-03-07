from dolfin import *
from femorph import *
import mshr
from dolfin_adjoint import *
from ufl import replace
import matplotlib.pyplot as plt
import numpy as np


def sympy_expr():
    import sympy as sp
    sppi, T, f, spx, spy  = sp.symbols("pi T f x[0] x[1]")
    p = 1./(pi*pi)*sp.sin(pi*spx)*sp.sin(pi*spy)
    T = -1./2*(-sp.diff(sp.diff(p, spx), spx) -sp.diff(sp.diff(p, spy), spy))
    f = 1.*(-sp.diff(sp.diff(T, spx), spx) -sp.diff(sp.diff(T, spy), spy))
    return str(p), str(T), str(f)

# Mesh and boundary markers
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class OuterBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0.0) or near(x[0],1)
                                or near(x[1],0) or near(x[1],1))

N = 200
mesh = UnitSquareMesh(N, N)
mesh = Mesh(mesh)
markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
markers.set_all(0)
obstacle = Boundary()
outer = OuterBoundary()
VariableBoundary = 1
FixedBoundary = 2
obstacle.mark(markers, VariableBoundary)
outer.mark(markers, FixedBoundary)


p_ex, t_ex, f_ex = sympy_expr()

f = Expression(f_ex, degree=3)

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S)
ALE.move(mesh, s)

# Setup
V = FunctionSpace(mesh, "CG", 3)
u, v = Function(V), Function(V)
J = u*u*dx
F = inner(grad(u), grad(v))*dx + u*v*dx - f*v*dx
L = J + F

# Solve State
State = derivative(L, v, TestFunction(V))
State = replace(State, {u: TrialFunction(V)})

# FIXME: Handling of Dirichlet Conditions
# bcState = DirichletBC(V, Constant(0.0), markers, VariableBoundary)
bcState = DirichletBC(V, Constant(0.0), "on_boundary")

T = Function(V, name="T")
# solve(lhs(State) == rhs(State), T)
# FIXME: Handling of Dirichlet Conditions
solve(lhs(State) == rhs(State), T, bcs=bcState)

J = replace(J, {u: T})
J = assemble(J)
Jhat = ReducedFunctional(J, Control(s))

n = VolumeNormal(mesh)
s2 = Function(S)
s2.vector()[:] = n.vector()[:]
s0 = Function(S)

taylor_test(Jhat, s0, s2, dJdm=0)
print("-"*10)
taylor_test(Jhat, s0, s2)

