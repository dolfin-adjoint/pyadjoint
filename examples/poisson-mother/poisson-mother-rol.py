from __future__ import print_function
from fenics import *
from fenics_adjoint import *
import numpy

set_log_level(ERROR)

n = 10
mesh = UnitSquareMesh(n, n)

"""
This is to showcase mesh independence:
if we do mesh refinement but choose the wrong inner product,
then the iteration count will blow up.
"""
inner_product = "L2"
numpy.random.seed(0)


def randomly_refine(initial_mesh, ratio_to_refine=.3):
    cf = CellFunction('bool', initial_mesh)
    for k in range(len(cf)):
        if numpy.random.rand() < ratio_to_refine:
            cf[k] = True
    return refine(initial_mesh, cell_markers=cf)


k = 4
for i in range(k):
    mesh = randomly_refine(mesh)

cf = CellFunction("bool", mesh)
subdomain = CompiledSubDomain(
    'std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
subdomain.mark(cf, True)
mesh = refine(mesh, cf)
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = interpolate(Expression("x[0]+x[1]", degree=1), W)
u = Function(V, name='State')
v = TestFunction(V)

F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)


x = SpatialCoordinate(mesh)
w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
d = 1/(2*pi**2)
d = Expression("d*w", d=d, w=w, degree=3)

alpha = Constant(1e-3)
J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = Control(f)

rf = ReducedFunctional(J, control)
params_dict = {
    'General': {
        'Secant': {
            'Type': 'Limited-Memory BFGS',
            'Maximum Storage': 5
        }
    },
    'Step': {
        'Type': 'Line Search',
        'Line Search': {
            'Descent Method': {
                'Type': 'Quasi-Newton Method'
            },
            'Curvature Condition': {
                'Type': 'Strong Wolfe Conditions'
            }
        }
    },
    'Status Test': {
        'Gradient Tolerance': 1e-10,
        'Step Tolerance': 1e-16,
        'Relative Step Tolerance': 1e-10,
        'Iteration Limit': 50
    }
}
problem = MinimizationProblem(rf)
solver = ROLSolver(problem, params_dict, inner_product=inner_product)
sol = solver.solve()
out = File("f.pvd")
out << sol
f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w",
                        w=w, alpha=alpha, degree=3)
print(errornorm(f_analytic, sol))

problem = MinimizationProblem(rf, bounds=(0, 0.5))
solver = ROLSolver(problem, params_dict, inner_product=inner_product)
sol = solver.solve()
out << sol
