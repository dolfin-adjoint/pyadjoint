.. _timestepping:

.. py:currentmodule:: dolfin_adjoint

======================================
A library for writing transient models
======================================

In this section, the experimental timestepping library is described.
The timestepping library offers **an alternative way of writing forward models**
that enables extra optimisations in the adjoint run. **dolfin-adjoint handles
time-dependent models without the use of this library.**

Transient models typically consist of a known repeating model "timestep". This
leads to a repeating model structure, and this structure may be exploited to
increase model performance. In particular, if the structure of the transient
model is known, it is possible for static data to be pre-computed and cached
before timestepping the model itself.

The dolfin-adjoint source code includes an additional **experimental** library,
known as the timestepping library, which enables such optimisations to be
performed. This library may be used on its own or in combination with the
dolfin-adjoint library. The library source code can be found in the
timestepping/ directory of the dolfin-adjoint source tree, and more complete
documentation can be found in the timestepping/manual/ directory.

******************************
The timestepping Python module
******************************

The timestepping library can be accessed via:

.. code-block:: python

  from dolfin import *
  from timestepping import *

This provides additional functionality enabling a transient model to be
described. For example, the following yields a very simple model for the
diffusion equation:

.. code-block:: python

  from dolfin import *
  from timestepping import *

  # Define a simple structured mesh on the unit interval
  mesh = UnitIntervalMesh(10)
  # P1 function space
  space = FunctionSpace(mesh, "CG", 1)

  # Model parameters and boundary conditions
  dt = StaticConstant(0.05)
  bc1 = StaticDirichletBC(space, 1.0,
    "on_boundary && near(x[0], 0.0)")
  bc2 = StaticDirichletBC(space, 0.0,
    "on_boundary && near(x[0], 1.0)")
  bcs = [bc1, bc2]
  nu = StaticConstant(0.01)

  # Define time levels
  levels = TimeLevels(levels = [n, n + 1], cycle_map = {n:n + 1})
  # A time dependent function
  u = TimeFunction(levels, space, name = "u")

  # Initialise a TimeSystem
  system = TimeSystem()

  # Add an initial assignment
  u_ic = StaticFunction(space, name = "u_ic")
  u_ic.assign(Constant(0.0))
  bc1.apply(u_ic.vector())
  system.add_solve(u_ic, u[0])
  # Register a simple diffusion equation, discretised in time
  # using forward Euler
  test = TestFunction(space)
  system.add_solve(
    inner(test, (1.0 / dt) * (u[n + 1] - u[n])) * dx ==
      -nu * inner(grad(test), grad(u[n])) * dx,
    u[n + 1], bcs,
    solver_parameters = {"linear_solver":"lu"})

  # Assemble the TimeSystem
  system = system.assemble()

  # Timestep the model
  t = 0.0
  while t * (1.0 + 1.0e-9) < 1.0:
    system.timestep()
    t += float(dt)
  # Finalise
  system.finalise()

The timestepping library can derive discrete adjoint models and perform
derivative calculations. Time discretisation specific optimisations are
applied to the adjoint model. The following modification to the above example
performs such a calculation, and verifies the computed derivative via a
Taylor remainder test:

.. code-block:: python

  # Assemble the TimeSystem, enabling the adjoint. Set the
  # functional to be equal to spatial integral of the final u.
  system = system.assemble(adjoint = True, functional = u[N] * dx)

  # Timestep the model
  t = 0.0
  while t * (1.0 + 1.0e-9) < 1.0:
    system.timestep()
    t += float(dt)
  # Finalise
  system.finalise()

  # Perform a total derivative calculation
  dJ = system.compute_gradient(nu)

  # Verify the stored forward model data
  system.verify_checkpoints()
  # Verify the computed derivative using a Taylor remainder
  # convergence test
  orders = system.taylor_test(nu, grad = dJ)
  # Check the convergence order
  assert((orders > 1.99).all())

*********************************************
The dolfin_adjoint_timestepping Python module
*********************************************

The functionality of the timestepping and dolfin-adjoint libraries can be
combined via:

.. code-block:: python

  from dolfin import *
  from dolfin_adjoint_timestepping import *

The following example constructs a very simple model for the diffusion equation
using the timestepping library. dolfin-adjoint is then used to derive a
discrete adjoint model, perform a total derivative calculation, and verify the
computed derivative:

.. code-block:: python

  from dolfin import *
  from dolfin_adjoint_timestepping import *

  ### Stage 1: Configure and execute the forward model using
  ###          functionality provided by the timestepping library

  # Define a simple structured mesh on the unit interval
  mesh = UnitIntervalMesh(10)
  # P1 function space
  space = FunctionSpace(mesh, "CG", 1)

  # Model parameters and boundary conditions
  dt = StaticConstant(0.05)
  bc1 = StaticDirichletBC(space, 1.0,
    "on_boundary && near(x[0], 0.0)")
  bc2 = StaticDirichletBC(space, 0.0,
    "on_boundary && near(x[0], 1.0)")
  bcs = [bc1, bc2]
  nu = StaticConstant(0.01)

  # Define time levels
  levels = TimeLevels(levels = [n, n + 1], cycle_map = {n:n + 1})
  # A time dependent function
  u = TimeFunction(levels, space, name = "u")

  # Initialise a TimeSystem
  system = TimeSystem()

  # Add an initial assignment
  u_ic = StaticFunction(space, name = "u_ic")
  u_ic.assign(Constant(0.0))
  bc1.apply(u_ic.vector())
  system.add_solve(u_ic, u[0])
  # Register a simple diffusion equation, discretised in time
  # using forward Euler
  test = TestFunction(space)
  system.add_solve(
    inner(test, (1.0 / dt) * (u[n + 1] - u[n])) * dx ==
      -nu * inner(grad(test), grad(u[n])) * dx,
    u[n + 1], bcs,
    solver_parameters = {"linear_solver":"lu"})

  # Assemble the TimeSystem
  system = system.assemble(initialise = False)

  # Run the forward model. The model execution is wrapped by a
  # function to enable adjoint verification using the
  # dolfin-adjoint taylor_test function.
  def run_forward():
    system.initialise()
    t = 0.0
    while t * (1.0 + 1.0e-9) < 1.0:
      system.timestep()
      t += float(dt)
    system.finalise()
    return
  run_forward()

  ### Stage 2: Access features provided by the dolfin-adjoint library

  # Disable annotation of model equations by dolfin-adjoint
  parameters["adjoint"]["stop_annotating"] = True

  # Define a functional equal to spatial integral of the final u
  J = u[N] * dx
  # Perform a total derivative calculation
  J_da = Functional(J * dolfin_adjoint.dt[FINISH_TIME])
  nu_da = Control(nu)
  dJ = compute_gradient(J_da, nu_da)

  # Verify the computed derivative using a Taylor remainder
  # convergence test
  def J_p(nu_p):
    nu.assign(nu_p)
    system.reassemble(nu)
    run_forward()
    return assemble(J)
  order = taylor_test(J_p, nu_da, assemble(J), dJ, seed = 1.0e-6)
  # Check the convergence order
  assert(order > 1.99)

The native timestepping Python module can often yield faster adjoint models than
the dolfin_adjoint_timestepping module, but is much less feature complete.
