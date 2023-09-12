===================================
Properties of the adjoint equations
===================================

The adjoint equations have a reputation for being
counterintuitive. When told the adjoint equations run backwards in
time, this can strike the novice as bizarre. Therefore, it is worth
taking some time to explore these properties, until it is *obvious*
that the adjoint system should run backwards in time, and (more
generally) reverse the propagation of information. In fact, these
supposedly confusing properties are induced by nothing more exotic
than simple transposition.

***************************************************
The adjoint reverses the propagation of information
***************************************************

A simple advection example
==========================

Suppose we are are solving a one-dimensional advection-type equation
on a mesh with three nodes, at :math:`x_0=0`, :math:`x_1=0.5`, and
:math:`x_2=1`.  The velocity goes from left to right, and so we impose
an inflow boundary condition at the left-most node :math:`x_0`. A
simple sketch of the linear system that might describe this
configuration could look as follows:

.. math::

  \begin{pmatrix} 1 & 0 & 0 \\
                  a & b & 0 \\
                  c & d & e \end{pmatrix}
  \begin{pmatrix} u_0 \\ u_1 \\ u_2 \end{pmatrix}
  =
  \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix},

where :math:`a, b, c, d` and :math:`e` are some coefficients of the
matrix arising from a discretisation of the equation. The equation for
:math:`u_1` does not depend on :math:`u_2`, as information is flowing
from left to right.  The structure of the matrix dictates the
propagation of information of the system: first :math:`u_0` is set to
the boundary condition value, then :math:`u_1` may be computed, and
then finally :math:`u_2`. The *lower-triangular nature of the matrix
reflects the rightward propagation of information.*

Notice that :math:`u_0` is *prescribed*: that is, the value of
:math:`u_0` does not depend on the values at any other nodes; all
off-diagonal entries on the row for :math:`u_0` are zero. Notice
further that the value :math:`u_2` is *diagnostic*: no other nodes
depend on its value; all off-diagonal entries on its column are zero.

Now suppose that we take the adjoint of this system with respect to
some functional :math:`J(u)`. The operator is linear (no entry in the
matrix depends on :math:`u`), and so the adjoint of this system is
just its transpose:

.. math::

  \begin{pmatrix} 1 & a & c \\
                  0 & b & d \\
                  0 & 0 & e \end{pmatrix}
  \begin{pmatrix} \lambda_0 \\ \lambda_1 \\ \lambda_2 \end{pmatrix}
  =
  \begin{pmatrix} {\partial J}/{\partial u_0} \\ {\partial J}/{\partial u_1} \\ {\partial J}/{\partial u_2} \end{pmatrix},

.. sidebar:: The adjoint of the advection equation

  If the forward equation is :math:`u \cdot \nabla T`, where :math:`u`
  is the advecting velocity and :math:`T` is the advected tracer, then
  its corresponding adjoint term is :math:`-u \cdot \nabla \lambda`.
  The adjoint advection equation is itself an advection equation, with
  the reverse of the forward velocity.

where :math:`\lambda` is the adjoint variable corresponding to
:math:`u`. Observe that transposing the forward system yields an
upper-triangular adjoint system: *the adjoint propagates information
from right to left, in the opposite sense to the propagation of the
forward system*. To solve this system, one would first solve for
:math:`\lambda_2`, then compute :math:`\lambda_1`, and finally
:math:`\lambda_0`.

Further notice that :math:`\lambda_2` is now prescribed: it can be
computed directly from the data, with no dependencies on the values of
other adjoint variables; all of the off-diagonal entries in its row
are zero. :math:`\lambda_0` is now diagnostic: no other variables
depend on its value; all off-diagonal entries in its column are zero.

.. sidebar:: Prescribed and diagnostic variables

  This is a general pattern. Variables that are prescribed in the
  forward model are diagnostic in the adjoint; variables that are
  diagnostic in the forward model are prescribed in the adjoint.

A time-dependent example
========================

Now consider a time-dependent system. For convenience, we assume the
system is linear, but the result holds true in exactly the same way
for nonlinear systems. We start with an initial condition :math:`f_0`
for :math:`u_0` (where the subscript denotes the timestep, rather than
the node). We then use this information to compute the value at the
next timestep, :math:`u_1`. This information is then used to compute
:math:`u_2`, and so on. This temporal structure can be represented as
a *block-structured* matrix:

.. math::

  \begin{pmatrix} I & 0 & 0 \\
                  A & B & 0 \\
                  C & D & E \end{pmatrix}
  \begin{pmatrix} u_0 \\ u_1 \\ u_2 \end{pmatrix}
  =
  \begin{pmatrix} f_0 \\ f_1 \\ f_2 \end{pmatrix},

where :math:`I` is the identity operator, :math:`A, B, C, D` and
:math:`E` are some operators arising from the discretisation of the
time-dependent system, :math:`f_0` is the initial condition for
:math:`u_0`, and :math:`f_n` is the source term for the equation for
:math:`u_n`.

Again, the *temporal propagation of information forward in time is
reflected in the lower-triangular structure of the matrix*. This
reflects the fact that it is possible to timestep the system, and
solve for parts of the solution :math:`u` at a time. If the discrete
operator were not lower-triangular, all timesteps of the solution
:math:`u` would be coupled, and would have to be solved for together.

Notice again that the value at the initial time :math:`u_0` is
prescribed, and the value at the final time :math:`u_2` is diagnostic.

Now let us take the adjoint of this system. Since the operator has
been assumed to be linear, the adjoint of this system is given by the
block-structured matrix

.. math::

  \begin{pmatrix} I & A^* & C^* \\
                  0 & B^* & D^* \\
                  0 & 0   & E^* \end{pmatrix}
  \begin{pmatrix} \lambda_0 \\ \lambda_1 \\ \lambda_2 \end{pmatrix}
  =
  \begin{pmatrix} {\partial J}/{\partial u_0} \\ {\partial J}/{\partial u_1} \\ {\partial J}/{\partial u_2} \end{pmatrix},

where :math:`\lambda` is the adjoint variable corresponding to
:math:`u`. Observe that the adjoint system is now upper-triangular:
*the adjoint propagates information from later times to earlier times,
in the opposite sense to the propagation of the forward system*. To
solve this system, one would first solve for :math:`\lambda_2`, then
compute :math:`\lambda_1`, and finally :math:`\lambda_0`.

Notice once more that the prescribed-diagnostic relationship
applies. In the forward model, the initial condition is prescribed,
and the solution at the final time is diagnostic. In the adjoint
model, the solution at the final time is prescribed (a so-called
*terminal condition*, rather than an initial condition), and the
solution at the beginning of time is diagnostic. This is why when the
adjoint of a continuous system is derived, the formulation always
includes the specification of a terminal condition on the adjoint
system.

******************************
The adjoint equation is linear
******************************

As noted in the previous section, the operator of the tangent linear
system is the linearisation of the operator about the solution
:math:`u`; therefore, the adjoint system is always linear in
:math:`\lambda`.

.. sidebar:: Unconverged nonlinear iterations

  Note that the nonlinear iteration *has to converge* for the
  linearisation about the solution at that timestep to be valid. If
  the model does not drive the nonlinear problem to convergence
  (perhaps it only does a fixed number of Picard iterations, say),
  then it is not consistent to see the nonlinear solve as one
  equation, and to trade it for a linear solve in the adjoint. In
  other words, if the nonlinear solve does not converge, then each
  iteration of the *unconverged* nonlinear solve induces a linear
  solve in the adjoint system, and so the adjoint will take
  approximately the same runtime as the forward model.

  Converging your nonlinear problem is not only more accurate, it
  makes the adjoint relatively much more efficient!

This has two major effects. The first is a beneficial effect on the
computation time of the adjoint run: while the forward model may be
nonlinear, *the adjoint is always linear, and so it can be much
cheaper to solve than the forward model*.  For example, if the forward
model employs a Newton solver for the nonlinear problem that uses on
average :math:`5` linear solves to converge to machine precision, then
a rough estimate for the adjoint computation is that it will take
:math:`1/5` the runtime of the forward model.

The second major effect is on the storage requirements of the adjoint
run. Unfortunately, this effect is not beneficial.  The adjoint
operator is a linearisation of the nonlinear operator about the
solution :math:`u`: therefore, *if the forward model is nonlinear, the
forward solution must be available to assemble the adjoint system*. If
the forward model is steady, this is not a significant difficulty:
however, *if the forward model is time-dependent, the entire solution
trajectory through time must be available*.

The obvious approach to making the entire solution trajectory
available is to store the value of every variable solved for. This
approach is the simplest, and it is the most efficient option if
enough storage is available on the machine to store the entire
solution at once. However, for long simulations with many degrees of
freedom, it is usually impractical to store the entire solution
trajectory, and therefore some alternative approach must be
implemented.

.. The space cost of storing all variables is linear in time (double the
.. timesteps, double the storage) and the time cost is constant (no extra
.. recomputation is required). The opposite strategy, of storing nothing
.. and recomputing everything when it becomes necessary, is quadratic in
.. time and constant in space. A *checkpointing algorithm* attempts to
.. strike a balance between these two extremes to control both the
.. spatial requirements (storage space) and temporal requirements
.. (recomputation).

.. Checkpointing algorithms have been well studied in the literature,
.. usually in the context of algorithmic differentiation
.. :cite:`griewank1992` :cite:`hinze2005` :cite:`stumm2010`
.. :cite:`wang2009`.  There are two categories of checkpointing
.. algorithms: *offline* algorithms and *online* algorithms.  In the
.. offline case, the number of timesteps is known in advance, and so the
.. optimal distribution of checkpoints may be computed a priori (and
.. hence "offline"), while in the online case, the number of timesteps is
.. not known in advance, and so the distribution of checkpoints must be
.. computed during the run itself. Of particular note is the revolve
.. software of Griewank and Walther, which achieves logarithmic growth of
.. both space *and* time :cite:`griewank2000`.  This algorithm is
.. provably optimal for the offline case :cite:`grimm1996`.

Summary
=======

Now that the adjoint and tangent linear equations have been
introduced, and some of their properties discussed, let us see in more
detail the applications of these concepts. This is discussed in
:doc:`the next section <5-applications>`.

.. .. rubric:: References

.. .. bibliography:: 4-adjoint.bib
..    :cited:
..    :labelprefix: 4M-
