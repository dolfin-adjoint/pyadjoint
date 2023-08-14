============================
Generalised stability theory
============================

************
Introduction
************

The stability of a physical system is a classical problem of
mechanics, with contributions from authors such as Lagrange, Dirichlet
and Lyapunov :cite:`leine2010`. Stability investigates the response of
the system to small perturbations in its initial condition: if the
solutions of the perturbed systems remain within a neighbourhood of
the unperturbed solution, then the system is stable; otherwise, the
system is unstable at that initial condition.

The traditional approach for investigating the stability of physical
systems was given by Lyapunov :cite:`lyapunov1892`. The (nonlinear)
equations of motion are linearised about a base solution, and the
eigenvalues of the linearised system are computed. If all eigenvalues
have negative real part, then there exists a finite region of
stability around the initial condition: perturbations within that
region decay to zero, and the system is asymptotically stable
:cite:`parks1992`.

While this approach has had many successes, several authors have noted
that it does not give a complete description of the finite-time
stability of a physical system. While the eigendecomposition
determines the asymptotic stability of the linearised equations as
:math:`t \rightarrow \infty`, some systems permit transient
perturbations which grow in magnitude, before being predicted to
decay. However, if the perturbations grow too large, the linearised
equations may cease to hold, and the system may become unstable due to
nonlinear effects. More specifically, this transient growth occurs
when the system is non-normal, i.e. when the eigenfunctions of the
system do not form an orthogonal basis :cite:`schmid2007`.  For
example, Trefethen :cite:`trefethen1993` describes how the traditional
approach fails to give accurate stability predictions for several
classical problems in fluid mechanics, and resolves the problem by
analysing the nonnormality of the system in terms of pseudospectra
:cite:`trefethen2006`.

Therefore, this motivates the development of a finite-time theory of
stability, to investigate and predict the transient growth of
perturbations. While Lorenz :cite:`lorenz1965` discussed the core
ideas (without using modern nomenclature), the development of this
so-called generalised stability theory (GST) has been driven by the
work of B. F. Farrell and co-workers :cite:`farrell1982a`
:cite:`farrell1985` :cite:`farrell1996` :cite:`farrell1996b`. The main
idea is to consider the linearised *propagator* of the system, the
operator (linearised about the time-dependent trajectory) that maps
perturbations in the initial conditions to perturbations in the final
state. Essentially, the propagator is the inverse of the tangent
linear system associated with the nonlinear forward model, along with
operators to load the initial perturbation and select the final
perturbation. The perturbations that grow maximally over the time
window are given by the singular functions of the propagator
associated with the largest singular values. Since the linearised
propagator depends on the base solution, it follows that the
predictability of the system depends on the conditions of the base
solution itself: some states are inherently more predictable than
others :cite:`lorenz1965` :cite:`kalnay2002`.

**************************************************
The singular value decomposition of the propagator
**************************************************

This presentation of generalised stability theory will consider the
stability of the system to perturbations in the initial conditions,
but the same approach can be applied to analysing the stability of the
system to perturbations in other parameters.

Consider the solution of the model at the final time :math:`u_T` as a
pure function of the initial condition :math:`u_0`:

.. math::

  u_T = M(u_0),

where :math:`M` is the *nonlinear propagator* that advances the
solution in time over a given finite time window :math:`[0, T]`.
Other parameters necessary for the solution (e.g. boundary conditions,
material parameters, etc.)  are considered fixed. Assuming the model
is sufficiently differentiable, the response of the model :math:`M` to
a perturbation :math:`\delta u_0` in :math:`u_0` is given by

.. math::

  \delta u_T = M(u_0 + \delta u_0) - M(u_0) = \frac{\textrm{d} M}{\textrm{d} u_0} \delta u_0 + O(\left|\left|\delta u_0\right|\right|^2).

Neglecting higher-order terms, the linearised perturbation to the
final state is given by

.. math::

  \delta u_T \approx L \delta u_0,

where :math:`L` is the *linearised propagator* (or just propagator)
:math:`{\textrm{d} M}/{\textrm{d} u_0}` that advances perturbations in
the initial conditions to perturbations to the final solution.

To quantify the stability of the system, we wish to identify
perturbations :math:`\delta u_0` that grow the most over the time
window :math:`[0, T]`.  For simplicity, equip both the initial
condition and final solutions with the conventional inner product
:math:`\left\langle \cdot, \cdot \right\rangle`.  We seek the initial
perturbation :math:`\delta u_0` of unit norm :math:`\left|\left|\delta
u_0\right|\right| = \sqrt{\left\langle \delta u_0, \delta u_0
\right\rangle} = 1` such that

.. math::

  \delta u_0 = \operatorname*{arg\,max}_{\left|\left|\delta u_0\right|\right|} \left\langle \delta u_T, \delta u_T \right\rangle.

Expanding :math:`\delta u_T` in terms of the propagator,

.. math::

  \left\langle \delta u_T, \delta u_T \right\rangle = \left\langle L \delta u_0, L \delta u_0 \right\rangle = \left\langle \delta u_0, L^*L \delta u_0 \right\rangle,

we see that the leading perturbation is the eigenfunction of
:math:`L^*L` associated with the largest eigenvalue :math:`\mu`, and
the growth of the norm of the perturbation is given by
:math:`\sqrt{\mu}`. In other words, the leading initial perturbation
:math:`\delta u_0` is the leading right singular function of
:math:`L`, the resulting final perturbation :math:`\delta u_T` is the
associated left singular function, and the growth rate of the
perturbation is given by the associated singular value
:math:`\sigma`. The remaining singular functions offer a similar
physical interpretation: if a singular function :math:`v` has an
associated singular value :math:`\sigma > 1`, the perturbation will
grow over the finite time window :math:`[0, T]`; if :math:`\sigma <
1`, the perturbation will decay over that time window.

If the initial condition and final solution spaces are equipped with
inner products :math:`\left\langle \cdot, \cdot \right\rangle_I \equiv
\left\langle \cdot, X_I \cdot \right\rangle` and :math:`\left\langle
\cdot, \cdot \right\rangle_F \equiv \left\langle \cdot, X_F \cdot
\right\rangle` respectively, then the leading perturbations are given
by the eigenfunctions

.. math::

  X_I^{-1} L^* X_F L \delta u_0 = \mu \delta u_0.

The operators :math:`X_I` and :math:`X_F` must be symmetric
positive-definite. In the finite element context, :math:`X_I` and
:math:`X_F` are often the mass matrices associated with the input and
output spaces, as these matrices induce the functional :math:`L_2`
norm.

************************
Computing the propagator
************************

In general, the nonlinear propagator :math:`M` that maps initial
conditions to final solutions is not available as an explicit
function; instead, a PDE is solved. For clarity, let :math:`m` denote
the data supplied for the initial condition. The PDE may be written in
the abstract implicit form

.. math::

  F(u, m) = 0,

with the understanding that :math:`u_0 = m`. We assume that for any
initial condition :math:`m`, the PDE can be solved for the solution
trajectory :math:`u`, and the nonlinear propagator :math:`M` can then
be computed by returning the solution at the final
time. Differentiating the PDE with respect to the initial condition
data :math:`m` yields

.. math::

  \frac{\partial F}{\partial u} \frac{\textrm{d}u}{\textrm{d}m} = - \frac{\partial F}{\partial m},

the *tangent linear system* associated with the PDE.  The term
:math:`{\partial F}/{\partial u}` is the PDE operator linearised about
the solution trajectory :math:`u`: therefore, it is linear, even when
the original PDE is nonlinear. :math:`{\partial F}/{\partial m}`
describes how the equations change as the initial condition data
:math:`m` changes, and acts as the source term for the tangent linear
system. :math:`{\textrm{d}u}/{\textrm{d}m}` is the prognostic variable
of the tangent linear system, and describes how the solution changes
with changes to :math:`m`. To evaluate the action of the propagator
:math:`L` on a given perturbation :math:`\delta m`, the tangent linear
system is solved with that particular perturbation, and evaluated at
the final time:

.. math::

  L \delta m \equiv - \left.\left(\frac{\partial F}{\partial u}\right)^{-1}\frac{\partial F}{\partial m} \delta m\right|_T.

Therefore, to automate the generalised stability analysis of a PDE, it
is necessary to automatically derive and solve the associated tangent
linear system. Furthermore, as the GST analysis also requires the
adjoint of the propagator, it is also necessary to automatically
derive and solve the adjoint of the tangent linear system. This is why
GST is considered as an application of adjoints.

**************************
Singular value computation
**************************

Once the propagator :math:`L` is available, its singular value
decomposition may be computed.  There are two main computational
approaches. The first approach is to compute the eigendecomposition of
the *cross product* matrix :math:`L^*L` (or :math:`LL^*`, whichever is
smaller). The second is to compute the eigendecomposition of the
*cyclic* matrix

.. math::

  H(L) =
  \begin{pmatrix} 0 & L \\
                L^* & 0
  \end{pmatrix}

As explained in :cite:`trefethen1997`, the latter option is more
accurate for computing the small singular values, but is more
expensive. As we are only interested in a small number of the largest
singular triplets, the cross product approach is used throughout this
work. Note that regardless of which approach is taken, the adjoint
propagator :math:`L^*` is necessary to compute the SVD of :math:`L`.

The algorithm used to compute the eigendecomposition of the cross
product matrix is the Krylov-Schur algorithm :cite:`stewart2001`, as
implemented in `SLEPc <http://www.grycap.upv.es/slepc/>`_
:cite:`hernandez2005` :cite:`hernandez2007b`. As the matrix is
Hermitian (whether norms are used or not), this algorithm reduces to
the thick-restart variant :cite:`wu2000` of the Lanczos method
:cite:`lanczos1950`.  This algorithm was found experimentally to be
faster than all other algorithms implemented in SLEPc for the
computation of a small number of singular triplets, which is the case
of interest in stability analysis.

Rather than representing the propagator as a matrix, the action of the
propagator is computed in a matrix-free fashion, using the tangent
linear model. In turn, the entire time-dependent tangent linear model
is not stored, but its action is computed in a global-matrix-free
fashion, using the matrices associated with each individual equation
solve.  In turn, the solution of each equation solve may optionally be
achieved in a matrix-free fashion; the automatic derivation of the
tangent linear and adjoint systems supports such an approach.
Similarly, the adjoint propagator is computed in a matrix-free fashion
using the adjoint model. SLEPc elegantly supports such matrix-free
computations through the use of PETSc shell matrices :cite:`balay2010`
:cite:`balay1997`.

.. For more information on how to perform a generalised stability
.. analysis with dolfin-adjoint, see the :doc:`chapter in the
.. documentation on generalised stability analysis
.. <../gst>`.

.. rubric:: References

.. bibliography:: A-gst.bib
   :cited:
   :labelprefix: AM-
