========================
Applications of adjoints
========================

As mentioned in the introduction, adjoints (and tangent linear models)
have many applications in many different areas of computational
science.  In this section, we aim to give a very brief overview of
each application in which adjoints are used, with the intent of
getting the basic idea across.

For each application, a very brief literature review is provided,
giving pointers to some key references which may be used to explore
the field. I make no claim that these reviews are comprehensive;
naturally, I am personally more familiar with some areas than
others. Contributions to this section are very welcome.

****************************
PDE-constrained optimisation
****************************

As discussed in the previous sections, adjoints form the core
technique for efficiently computing the gradient
:math:`{\mathrm{d}J(u, m)}/{\mathrm{d}m}` of a functional :math:`J` to
be minimised.  This is usually essential for solving such optimisation
problems in practice: gradient-free optimisation algorithms typically
take orders of magnitude more iterations to converge; since each
iteration involves a PDE solve, minimising the number of iterations
taken is crucial.

For an engineering introduction to PDE-constrained optimisation,
Gunzburger's book is excellent :cite:`gunzburger2003`; for an in-depth
mathematical analysis, see the rigorous treatment of Hinze et
al. :cite:`hinze2009`. PDE-constrained optimisation is also referred
to in the literature as optimal control: the book of Lions
:cite:`lions1971` was a fundamental early contribution.

********************
Sensitivity analysis
********************

Occasionally, the gradient of a functional :math:`J` with respect to
some parameter :math:`m` is not merely required as an input to an
optimisation algorithm, but rather is of scientific interest in its
own right. Adjoint computations can tease apart hidden influences and
teleconnections; such computations can also inform scientists
regarding which variables matter the least, which is often important
for deriving approximate models; parameters with little impact on the
simulation can be ignored. This process is also often undertaken in
advance of solving an optimisation problem: by discarding parameters
which do not significantly influence the functional, the dimension of
the parameter space may be systematically reduced.

A fundamental early contribution was the work of Cacuci
:cite:`cacuci1981`. Much excellent work in applying sensitivity
analysis to questions of enormous scientific importance has been done
in the areas of oceanography and meteorology: partly because ocean and
atmospheric models often have adjoint versions implemented for the
purposes of data assimilation, partly because adjoint analysis is
often the only practical way of identifying such connections, and
partly because practitioners in these fields are aware of adjoints and
their potential. Of particular note is the work done by Heimbach and
co-workers using the adjoint of the `MITgcm ocean model
<http://mitgcm.org>`_ :cite:`losch2007` :cite:`heimbach2010`
:cite:`heimbach2012`.

*****************
Data assimilation
*****************

A forward model requires data on which to operate. For example, to
start a weather forecast, knowledge of the entire state of the
atmosphere at some point in time is required as an initial condition
from which to begin the simulation: start from the wrong initial
condition, and you will get the wrong weather.

The problem is that, in practice, the initial condition is
unknown. Instead, observations of the state of the atmosphere are
available, some available at the initial time, and some taken at later
times. The goal of data assimilation is to systematically combine
observations and computations, usually with the intention of acquiring
the best possible estimate of the unknown initial condition, so that a
forecast may be run. Indeed, most of the dramatic increase in forecast
skill over the past decade has been attributable to improvements in
data assimilation algorithms (Prof. Dale Barker, UK Met Office,
personal communication). This problem is routinely tackled in every
meteorological centre in the world, multiple times a day: your weather
forecast relies entirely upon it.

There are two major approaches to data assimilation. In a sequential
algorithm, the forward system is timestepped until an observation is
available, at which point the model is instantaneously "updated" to
incorporate the information contained in the observation. The most
popular approach to computing the amplitude of the update is the
Kalman filter algorithm :cite:`kalman1960`. The model is then
continued from this updated state until all of the observations are
used. A significant drawback of this approach is that an observation
only influences the state at times later than the observation time: in
other words, its temporal influence only propagates forward, not
backwards in time. This is a major disadvantage in the case where the
initial condition is the main quantity of interest, and most of the
observations are made towards the end of the time window, as is the
case in studies of mantle convection :cite:`bunge2003`.

The other major approach is referred to as variational data
assimilation, which is a special case of PDE-constrained
optimisation. In this approach, a functional :math:`J` is chosen to
represent the misfit between the observations and the computations,
weighted by the uncertainties in each. The initial condition is
treated as a control parameter :math:`m`, and chosen to minimise the
misfit :math:`J`.  The data assimilation community tends to place
significant emphasis on modelling the uncertainties in both the
computations and observations, as this is key to extracting the
maximal amount of information out of both.

.. sidebar:: The ECCO2 project

  One of the most impressive computational experiments ever attempted
  is the `ECCO2 project <http://ecco2.org>`_, which uses the adjoint
  of the `MITgcm ocean model <http://mitgcm.org>`_
  :cite:`heimbach2005` to assimilate every observation made of the
  world's oceans over the past twenty years :cite:`wunsch1996`. With
  this assimilation, they have produced the most accurate estimation
  of the state of the world's oceans ever devised. For a stunningly
  beautiful visualisation of this experiment, see
  `http://vimeo.com/39333560 <http://vimeo.com/39333560>`_.

Fundamental early works in field of variational data assimilation were
undertaken by Le Dimet and Talagrand :cite:`ledimet1986` and Talagrand
and Courtier :cite:`talagrand1987`. For an excellent introduction, see
the book of Kalnay :cite:`kalnay2002`. Information on the data
assimilation schemes used in practice by the `European Centre for
Medium-range Weather Forecasting
<http://www.ecmwf.int/research/ifsdocs/ASSIMILATION/Chap1_Overview2.html>`_
and the `UK Met Office
<http://www.metoffice.gov.uk/research/weather/data-assimilation-and-ensembles>`_
is available online.

****************
Inverse problems
****************

Data assimilation can be seen as a particular kind of inverse problem,
where the focus is on obtaining the best estimate for the system state
at some point in the past. More general inverse problems, where we
seek to gain information about unobservable system parameters from
observable system outputs, are ubiquitous in science and
engineering. Again, the same idea of minimising some functional that
measures the misfit between the observations and computed model
outputs plays a role. The field also has a heavy emphasis on
regularisation of inverse problems (which are generally ill-posed) and
on statistical estimates of uncertainty in the obtained results. For
an introductory textbook, see the book by Tarantola
:cite:`tarantola2005`; for a review of the current state of the art in
computational practice, see the compendium of Biegler et
al. :cite:`biegler2011`.

****************************
Generalised stability theory
****************************

The stability of solutions of physical systems is obviously of key
importance in many fields. The traditional approach to stability
theory is to linearise the operator about some state, and investigate
its eigenvalues: if the real component of every eigenvalue is
negative, the state is stable, and the associated eigenmode will
vanish in the limit as :math:`t \rightarrow \infty`; while if any
eigenvalue has a positive real part, the state is unstable, and the
associated eigenmode will grow in amplitude. While this traditional
approach works well in many cases, there are many important cases
where this analysis predicts stability where in fact the physical
system is unstable; in particular, this analysis fails when the
operator is *nonnormal* :cite:`trefethen1993` :cite:`schmid2007`.

.. sidebar:: Nonnormal matrices

  A matrix is normal if its eigenvectors form an orthonormal basis. A
  matrix is nonnormal if the eigenvectors have nonzero projection onto
  each other. See `the Wikipedia entry
  <http://en.wikipedia.org/wiki/Normal_matrix>`_ for more details.

In the nonnormal case, the usual stability theory fails. The two main
theoretical responses to this development have been the concepts of
pseudospectra (by Trefethen et al. :cite:`trefethen2006`) and
generalised stability theory (by Farrell et
al. :cite:`farrell1996`). Instead of focusing on the eigenvalues of
the operator linearised about some steady state, generalised stability
theory analyses the generalised eigenvalues associated with the
*propagator of the system*, which maps perturbations in initial
conditions to perturbations in the final state.  Essentially, the
propagator is the inverse of the tangent linear operator. By examining
these values, such an analysis can describe and predict the
perturbations that will grow maximally over *finite time windows*
:cite:`lorenz1965`. In order to compute these generalised eigenvalues
of the system propagator, both the tangent linear and adjoint
operators must be repeatedly solved :cite:`trefethen1997`.

As generalised stability theory yields information about the
perturbation directions which grow the most over the period of
interest, these vectors are often used to initialise ensemble members
to gain the optimal amount of information possible about the variance
of the ensemble :cite:`ioannou2006` :cite:`buizza2005`.  The growth
rates associated with these optimal perturbations have important
implications for the timescales of predictability of the physical
system. For examples of this analysis, see the work of Zanna et
al. :cite:`zanna2011b`.

We also note in passing that it is possible to use these singular
vectors to *guide* the targeting of observations to maximise the
effectiveness of a data assimilation strategy. For more details, see
:cite:`palmer1998`.

For more details, see the appendix on :doc:`generalised stability
theory <A-gst>`.

****************
Error estimation
****************

Another major application of adjoints is goal-based error estimation,
and the related computational technique of goal-based adaptivity. For
the purposes of this section, let :math:`u` and :math:`\lambda` denote
the *exact* forward and adjoint solutions associated with the PDE
:math:`F(u) = 0`, and let :math:`u_h` and :math:`\lambda_h` be some
approximations to them computed using a Galerkin finite element
discretisation. The fundamental question of goal-based error
estimation is: *what impact does the discretisation error* :math:`u -
u_h` *have on the error in the goal functional* :math:`J(u) - J(u_h)`?
One can construct cases where :math:`u - u_h` is large, but
:math:`J(u) - J(u_h)` is zero; similarly, one can construct cases
where :math:`u - u_h` is small, but :math:`J(u) - J(u_h)` is large.

The fundamental theorem of error estimation, due to Rannacher and
co-workers :cite:`becker2001` :cite:`bangerth2003`, states that

.. sidebar:: Residuals

  To compute the forward residual :math:`\rho_u`, take the approximate
  forward solution :math:`u_h` and plug it in to the forward equation
  :math:`\rho_u \equiv F(u_h)`. If :math:`u_h` were the exact
  solution, :math:`F(u_h)` would be zero, but since the solution is
  only approximate :math:`\rho_u \equiv F(u_h)` will be nonzero.

  To compute the adjoint residual, perform the analogous computation:
  take the approximate adjoint solution :math:`\lambda_h` and plug it
  in to the adjoint equation, and take all terms in the adjoint
  equation to the left-hand side.

.. math::

  J(u) - J(u_h) = \frac{1}{2} \left\langle \lambda - \lambda_h, \rho_u \right\rangle + \frac{1}{2} \left\langle u - u_h, \rho_{\lambda} \right\rangle + R_h^{(3)},

where :math:`u - u_h` is the discretisation error in the forward
solution, :math:`\lambda - \lambda_h` is the discretisation error in
the adjoint solution, :math:`\rho_u` is the forward residual,
:math:`\rho_{\lambda}` is the adjoint residual, and :math:`R_h^{(3)}`
is a remainder term which is *cubic* in the discretisation errors
:math:`u - u_h` and :math:`\lambda - \lambda_h`.

In practice, :math:`u - u_h` is estimated by approximating :math:`u`
with an extrapolation of :math:`u_h` into a higher-order function
space (and similarly for :math:`\lambda`), and the expression for
:math:`J(u) - J(u_h)` is broken up into a sum of element-level *error
indicators* that are used to decide which elements should be refined
in an adaptive algorithm.  For a discussion of how to implement
goal-based adaptivity in a general way in the FEniCS framework, see
the work of Rognes and Logg :cite:`rognes2010`.

.. sidebar:: The structure of the error estimator

  Notice that the error estimator has a very particular structure: it
  is the average of the inner product of the adjoint solution error
  with the forward residual, and the forward solution error with the
  adjoint residual. As many early results in the field only employed
  the first term in the error estimator, the approach became known as
  the "dual-weighted residual" approach (the term "dual" is commonly
  used to refer to the adjoint in this branch of the literature).

  If the averaging is not performed, and only the first term of the
  error estimator is included, the remainder term is *quadratic* in
  the forward and adjoint discretisation errors, not cubic.

The works of Rannacher and co-workers give many examples where a
computation that employs goal-based adaptivity is dramatically faster
at computing the functional to within a certain tolerance than the
corresponding fixed-mesh or heuristically-driven adaptivity. This
theorem raises the possibility of *reliable automated computation*:
not only can the discretisation of the differential equation be
automated with the FEniCS system, it can be automated to *reliably and
efficiently compute desired quantities to within a specified
accuracy*. The prospect of such a system would dramatically change the
social utility of computational science.

.. rubric:: References

.. bibliography:: 5-applications.bib
   :cited:
   :labelprefix: 5M-
