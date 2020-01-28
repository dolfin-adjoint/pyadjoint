#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _tube-shape-derivative-example:
#
# .. py:currentmodule:: dolfin_adjoint
#
# Tube shape-derivatives
# ======================
#
# .. sectionauthor:: Simon W. Funke <simon@simula.no>, Jørgen Dokken <jdokken@simula.no>, Stephan Schmidt <stephan.schmidt@mathematik.uni-wuerzburg.de>
#
# This example demonstrates how to compute shape-derivatives in morphing domains (including tube shape-derivatives)
# using dolfin-adjoint.

# Problem definition
# ******************
#
# We consider the problem of computing the sensitivity of the goal functional $J$
#
# .. math::
#       \min_{u, \Omega(t), t<0<T} \ J(u, \Omega) := \int_{0}^{T} \int_{\Omega(t)} \nabla u : \nabla u~\textrm{d}x\textrm{d}t
#
# where :math:`u` is the solution of the advection-diffusion equation in morphing domain:
#
# .. math::
#       u_t -k \Delta u - \nabla \cdot (u X_t) = 0  \qquad \mathrm{in}~\Omega(t)~\mathrm{for}~ 0<t<T
#
# The advection velocity is the domain morphing velocity (i.e. the velocity of the mesh nodes) and denoted as :math:`X_t`.
# In this example, the morphing domain  :math:`\Omega(t)` consists of a circle with a rotating cylindric hole.
# Since this geometry is rotational symmetric, we can avoid for mesh-deformation methods by
# creating a cylindric mesh with a hole and rotating the mesh over time:
#
# .. youtube:: jzUXfyFTDf4
#
# We note at this point, that dolfin-adjoint is not restricted to rotational symmetric geometries, but also computes the shape-derivatives in more general cases.
#
# On the boundaries of the outer and inner circles we enforce the conditions
#
# .. math::
#           u &= 1  \qquad \mathrm{on} \ \partial \Omega_{\textrm{inner}}~\mathrm{for}~ 0<t<T \\
#           \nabla u \cdot n + u X_t \cdot n &= 0  \qquad \mathrm{on} \ \partial \Omega_{\textrm{outer}}~\mathrm{for}~ 0<t<T \\

# The resulting shape gradient and state solution will look like this:

# .. youtube:: GymyW8MMD6A

#
# Implementation
# **************
#
# First, the :py:mod:`dolfin` and :py:mod:`dolfin_adjoint` modules are imported.
# We also reduce the log level for convenience and import pprint for prettier printing.

from dolfin import *
from dolfin_adjoint import *
from pprint import pprint
set_log_level(LogLevel.ERROR)

# Next, we load the mesh. The mesh was generated with gmsh; the source
# files are in the mesh directory.

fout = File("output/u.pvd")
mesh = Mesh("mesh/cable1.xml")
bdy_markers = MeshFunction("size_t", mesh, "mesh/cable1_facet_region.xml")

# Then, we define the discrete function spaces. A piecewise linear
# approximation is a suitable choice for the the solution of the advection-diffusion equation.
# In addition, we need a vector function space for the mesh deformations that describe the
# mesh morphing at every timestep. For this, a piecewise linear finite element space
# is the correct choice, since it has one degree of freedom at every mesh node.

V = VectorFunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)

# Next we define some important model parameters

k = Constant(0.01)       # Diffusion coefficient
omega = Constant(0.25)  # Rotation velocity
T = 4.0                 # Final time
dt = Constant(0.01)     # Time-step
N = int(T/float(dt))    # Number of timesteps
J = 0                   # Objective functional

# Next, we derive the weak variational form for computing the mesh coordinates
# at every timestep. We consider a rotating domain, hence the mesh coordinates follow
# the equation:
#
# .. math::
#     X_t = rot(X)
#
# with the mesh coordinates :math:`X=(x, y)` and the rotational velocity
# :math:`rot(X)=(2 \pi \omega y, -2 \pi \omega x)^T`
#
# Discretising this equation with a Crank-Nicolson scheme in time yields:
# Find :math:`S^n:=X^n-X^{n-1}` such that for all test functions :math:`z`
#
# .. math::
#     \left<S^n, z\right>_\Omega = \frac{1}{2} \Delta t (\left<rot(X^{n-1}+S^n)+rot(X^{n-1}), z\right>_\Omega),
#
# where the super-scripts denote the timelevels.
#
# In code, this becomes:

s = TrialFunction(V)
z = TestFunction(V)
S = Function(V)
X = SpatialCoordinate(mesh)
rot = lambda y: 2*pi*omega*as_vector((y[1], -y[0]))
F_s = lambda thn: inner(thn, z)*dx\
      - dt*0.5*inner(rot(X+thn)+rot(X), z)*dx

# In the time-loop, the solution :math:`S^n` will be used to update the mesh coordinates for the next time-level.
#
# Next, we derive the standard weak variational form for the diffusion-convection equation.
# We integrate the diffusion and advection term by parts in order to weakly enforce the
# boundary conditions on the outer circle. This yields: Find :math:`u` such that for all test
# functions :math:`v`
#
# .. math::
#     \left<u_t, v\right>_\Omega + k \left<\nabla (u), \nabla (v)\right>_\Omega + \left<X_t u, \nabla v \right>_{\Omega} = 0
#
# Discretising this equation in time using a Crank-Nicolson scheme yields the fully discretised problem:
# Find :math:`u^n` such that for all test
# functions :math:`v`
#
# .. math::

#     F_u(u^n, u^{n-1}, S^n;v) =&\  \frac{1}{\Delta t}\left<u^n-u^{n-1}, v\right>_\Omega \\
#             &\ + k \left<\nabla u^{n+1/2}, \nabla v\right>_\Omega \\
#             &\ + \left<X_t^{n+1/2} u^{n+1/2}, \nabla v \right>_{\Omega}
#            =&\ 0
#
# where the super-scripts denote the timelevel and the intermediate timelevels are defined as :math:`u^{n+1/2}:=\frac{1}{2} u^n + \frac{1}{2} u^{n-1}` and
# the mesh morphing velocity at :math:`X_t` at the intermediate timestep is approximated as :math:`X_t^{n+1/2}=\frac{1}{2} X_t^n + \frac{1}{2} X_t^{n-1} \approx \frac{1}{2\Delta t} S^n + \frac{1}{2} rot(X)`.
#
# In code, this becomes:

u0 = Function(W)
u1 = Function(W)
v = TestFunction(W)
w = TrialFunction(W)
F_u = lambda V: (1.0/dt*(w-u0)*v*dx
                  + k*inner(grad(v),Constant(1/2)*(grad(w)+grad(u0)))*dx
                  + inner(Constant(1/2)*(w+u0)*V, grad(v))*dx)

# Next, we define the Dirichlet boundary condition on the inner circle. The inner boundary edges are already marked
# in the mesh, so this is achieved with:

bc = DirichletBC(W, Constant(1.0), bdy_markers, 2)


# Next, we define the set of deformation functions.
# These functions will store the mesh coordinates changes
# from one timestep to the next and will be solved
# using the mesh deformation PDE above. Hence,
# we need as many deformation functions as there are
# timesteps in model (N). Later, we compute the derivative with respect
# to these variables with dolfin-adjoint.

thetas = [Function(V) for i in range(N+1)]

# The mesh movement per time-step is decomposed into a static component (mesh rotation) and a dynamic component (the control variables in thetas).
# The create a function which should contain the total movement per time-step, and assign the first control variable to it, assuming that the system starts from a static position.

S_tot = [Function(V) for i in range(N+1)]
S_tot[0].assign(thetas[0])

# Now we can implement the timeloop. It consist of four main steps:
#
# 1. Solve the mesh deformation PDE to compute the changes in mesh coordinates. During the shape-derivative step and add the control variable to the movement.
# 2. Update the mesh coordinates (using `ALE.move`);
# 3. Solve the advection-diffusion PDE;
# 4. Compute the contribution to the objective functional.
#
# The code is as follows:

ALE.move(mesh, S_tot[0])

for i in range(N):
    print("t=%.2f"%(float(i*dt)))

    # Solve for the fixed mesh displacement and assign this movement
    # summed with the control movement to the movement vector
    a, L = system(F_s(s))
    solve(a==L, S)
    S_tot[i+1].assign(S + thetas[i+1])

    # Move mesh
    ALE.move(mesh, S_tot[i+1])


    # Solve for state
    a, L = system(F_u(0.5/dt*(S_tot[i]+S_tot[i+1])))
    solve(a==L, u1, bc)
    u0.assign(u1)
    fout << u1

    # Compute functional
    J += assemble(dt*inner(grad(u1), grad(u1))*dx)

# This concludes the forward model, and we can now focus on computing the shape derivatives.
# As a first step, we define the control variables and the reduced functional. The control
# variables are the mesh deformation functions for all timesteps:

S_ctrls = thetas
ctrls = [Control(s) for s in S_ctrls]
Jhat = ReducedFunctional(J, ctrls)

# Now, we can run a Taylor test to verify the correctness of the shape derivatives and shape Hessian that dolfin-adjoint
# computes. The Taylor test performs a Taylor expansion in a user-specified perturbation direction.
# Since we have N control functions, we also need to specify N perturbation directions:


perbs = [project(0.01*Expression(["1-x[0]*x[0]-x[1]*x[1]", "1-x[0]*x[0]-x[1]*x[1]"], degree=2), V) for _ in ctrls]
conv = taylor_to_dict(Jhat, S_ctrls, perbs)
pprint(conv)

# Finally, we store the shape derivative for visualisation:

dJdm = Jhat.derivative()
ALE.move(mesh, Function(V), reset_mesh=True)

output = File("output/dJdOmega.pvd")
out = Function(V)
for s, dj in zip(S_tot, dJdm):
    ALE.move(mesh, s)
    out.assign(dj)
    output << out


# The example code can be found in ``examples/tube-shape-derivative`` in
# the ``dolfin-adjoint`` source tree, and executed as follows:
#
#
# .. code-block:: bash
#
#   $ python tube-shape-derivative.py
#
#     ...
#     {'R0': {'Rate': [0.99233621799267857, 0.9961586867939265,
#                      0.99807699260441085],
#             'Residual': [0.5980325614382096,
#                          0.3006089201505162,
#                          0.15070519330050303,
#                          0.07545310314151976]},
#      'R1': {'Rate': [1.99381570097023, 1.996905781695665, 1.9984531246419335],
#             'Residual': [0.00639804263023247,
#                          0.0016063818837048216,
#                          0.0004024577166074905,
#                          0.00010072236703549675]},
#      'R2': {'Rate': [2.9965354427868878, 2.998667114984213,
#                      2.9994355259119714],
#             'Residual': [5.510229348307336e-05,
#                          6.904347224064239e-06,
#                          8.638411247309367e-07,
#                          1.0802239755861346e-07]},
#      'eps': [0.01, 0.005, 0.0025, 0.00125]}


# The output shows the expected convergence rate for the finite difference (FD) test, first order adjoint test (dJdm),
# and second order adjoint test.
