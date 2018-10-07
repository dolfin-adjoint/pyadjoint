""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2

    subject to

    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega


"""

from dolfin import *
from dolfin_adjoint import *

set_log_level(LogLevel.ERROR)

# Create mesh, refined in the center
n = 16
mesh = Mesh(UnitSquareMesh(n, n))

cf = MeshFunction("bool", mesh, mesh.geometry().dim())
subdomain = CompiledSubDomain('std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
subdomain.mark(cf, True)
mesh = refine(mesh, cf)

# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = interpolate(Expression("0.0", degree=1, name='Control'), W)
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define functional of interest and the reduced functional
x = SpatialCoordinate(mesh)
w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3) 
d = 1/(2*pi**2)
d = Expression("d*w", d=d, w=w, degree=3)

alpha = Constant(1e-6)
J = assemble((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = Control(f)
rf = ReducedFunctional(J, control)
f_opt = minimize(rf, bounds=(0.0, 0.8), tol=1e-10,
                 options={"gtol": 1e-10, "factr": 0.0})

# Define the expressions of the analytical solution
f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w", w=w, alpha=alpha, degree=3)
u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)

f.assign(f_opt)
solve(F == 0, u, bc)
control_error = errornorm(f_analytic, f_opt)
state_error = errornorm(u_analytic, u)
print("h(min):           %e." % mesh.hmin())
print("Error in state:   %e." % state_error)
print("Error in control: %e." % control_error)

# Write solutions to XDMFFile, can be visualized with paraview
# First time step is approximated solution, second timestep is analytic
# solution
import os; os.system("rm output/*_scipy.*")
out_f = XDMFFile("output/f_scipy.xdmf")
out_f.write_checkpoint(f_opt, "f", 0.0, XDMFFile.Encoding.HDF5, True)
out_f.write_checkpoint(interpolate(f_analytic, W), "f",
                        1.0, XDMFFile.Encoding.HDF5, True)
out_u = XDMFFile("output/u_scipy.xdmf")
out_u.write_checkpoint(u, "u", 0.0, XDMFFile.Encoding.HDF5, True)
out_u.write_checkpoint(interpolate(u_analytic, V), "u",
                        1.0, XDMFFile.Encoding.HDF5, True)
