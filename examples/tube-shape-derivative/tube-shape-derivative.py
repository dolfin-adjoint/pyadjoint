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

k = Constant(0.1)    # Diffusion coefficient 
omega = 2            # Rotation velocity
T = 1.0              # Final time
dt = Constant(0.01)  # Timestep
N = int(T/float(dt)) # Number of timesteps
J = 0                # Objective functional

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

z = TestFunction(V)
S = Function(V)
X = SpatialCoordinate(mesh)
rot = lambda y: 2*pi*omega*as_vector((y[1], -y[0]))
F_s = inner(S, z)*dx - dt*0.5*inner(rot(X+S)+rot(X), z)*dx

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
F_u = lambda s1: \
      1.0/dt*w*v*dx \
    + k*inner(grad(v), 0.5*(grad(w)+grad(u0)))*dx \
    + inner(0.5*(w+u0)*0.5*(s1/dt+rot(X)), grad(v))*dx\
    - 1.0/dt*u0*v*dx

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

S_perb = [Function(V) for i in range(N+1)]

# Now we can implement the timeloop. It consist of four main steps:
#
# 1. Solve the mesh deformation PDE to compute the changes in mesh coordinates. During the shape-derivative step, 
#    we compute the derivative with respect to the outcome of this deformation. Hence, we use the `stop_annotating` 
#    command to tell dolfin-adjoint to ignore this solve and instead consider the PDE solution as a prognostic variable;
# 2. Update the mesh coordinates (using `ALE.move`);
# 3. Solve the advection-diffusion PDE;
# 4. Compute the contribution to the objective functional.
#
# The code is as follows:

ALE.move(mesh, S_perb[0])

for i in range(N):
    print("t=%.2f"%(float(i*dt)))

    # Solve for mesh displacement
    # Uncomment the next line to compute the tube-derivative (i.e. the ODE for mesh movement is part of the state equations).
    with stop_annotating():
        solve(F_s==0, S)

        # Move mesh
        ALE.move(mesh, S)
    ALE.move(mesh, S_perb[i+1])

    # Solve for state
    a, L = system(F_u(S_perb[i+1]))
    solve(a==L, u1, bc)
    u0.assign(u1)
    fout << u1

    # Compute functional
    J += assemble(dt*inner(grad(u1), grad(u1))*dx)

# This concludes the forward model, and we can now focus on computing the shape derivatives.
# As a first step, we define the control variables and the reduced functional. The control
# variables are the mesh deformation functions for all timesteps:

S_ctrls = S_perb
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
for s, dj in zip(S_ctrls, dJdm):
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
#    {'FD': {'Rate': [0.9056737635905056, 0.9549830726305613, 0.9779933598465425],
#            'Residual': [0.006199765976202087,
#                         0.0033093319122752263,
#                         0.0017071110639967912,
#                         0.0008666753413546502]},
#   'dJdm': {'Rate': [1.9985569243718035, 1.9993942922879513, 1.9997269353859186],
#            'Residual': [0.0008386349725824187,
#                         0.00020986856211702674,
#                         5.2489173199335264e-05,
#                         1.3124777243413068e-05]},
#   'Hm': {'Rate': [3.18526277864614, 3.105134801158437, 3.056300474150642],
#          'Residual': [1.4980232883393067e-06,
#                       1.6468685066275144e-07,
#                       1.913904258711021e-08,
#                       2.3008170675253208e-09]},
#   'eps': [0.01, 0.005, 0.0025, 0.00125]}



# The output shows the expected convergence rate for the finite difference (FD) test, first order adjoint test (dJdm),
# and second order adjoint test.

