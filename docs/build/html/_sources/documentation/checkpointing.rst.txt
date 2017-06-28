=============
Checkpointing
=============

As discussed in the :doc:`mathematical background <maths/index>`,
the adjoint model is a *linearisation* of the forward model. If the
forward model is nonlinear, then the solution of that forward model
must be available to *linearise* the forward model. By default,
dolfin-adjoint stores every variable computed in memory, as this is
the fastest and most straightforward option; however, this may not be
feasible for large runs, or for runs with very many timesteps.

The solution to this problem is to employ a *checkpointing
scheme*. Rather than store every variable during the forward run,
checkpoints are stored at strategically chosen intervals, from which
the model may recompute the missing solutions. During the adjoint run,
if a forward variable is necessary and unavailable, the forward model
is restarted from the nearest available checkpoint to compute the
missing solutions; once these are available, the adjoint run
continues.

Thus, to employ a checkpointing scheme, the control flow of the
adjoint run must seamlessly jump between assembling and solving the
adjoint equations, and assembling and solving parts of the forward
run. Coding a checkpointing scheme is quite complicated, and so most
hand-coded adjoint models do not use them. However, the
:doc:`libadjoint library underlying dolfin-adjoint <../citing/index>`
embeds the excellent `revolve library`_ of `Griewank and Walther`_,
and can automatically employ optimal checkpointing schemes for almost
no marginal user effort.

.. _revolve library: http://www2.math.uni-paderborn.de/index.php?id=12067&L=1
.. _Griewank and Walther: http://dx.doi.org/10.1145/347837.347846

Activating checkpointing is very straightforward: two calls to
dolfin-adjoint functions are necessary. Firstly, before any equations
are solved, the user must call the :py:func:`adj_checkpointing
<dolfin_adjoint.adj_checkpointing>` function, which activates and
configures the checkpointing scheme.  Secondly, the user must place a
call to :py:func:`adj_inc_timestep <dolfin_adjoint.adj_inc_timestep>`
at the end of the time loop, which indicates to libadjoint that a
timestep has ended. (Internally, the checkpointing scheme relies on
the concept of timesteps, but dolfin-adjoint has no way of
automatically determining when a timestep has ended, and so the user
must help out.) For example, to activate checkpointing for the
Burgers' equation:

.. literalinclude:: ../_static/tutorial5.py
    :emphasize-lines: 4,5,32

|more| Download the `checkpointed adjoint code`_.

.. _checkpointed adjoint code: ../_static/tutorial5.py


The adjoint is :doc:`still correct <verification>`:

.. code-block:: none

    $ python tutorial5.py
    ...
    Convergence orders for Taylor remainder with adjoint information (should all be 2):
      [1.9581779061731224, 1.9787032981594719, 1.9892527501829258, 1.994601330422228]

To see what the checkpointing scheme does, pass :py:data:`verbose=True`:

.. code-block:: none

    $ python tutorial5.py | grep Revolve
    Revolve: Checkpoint statistics:
    Revolve: Checkpoint timestep 0 on disk.
    Revolve: Advance from timestep 0 to timestep 3.
    Revolve: Checkpoint timestep 3 on disk.
    Revolve: Advance from timestep 3 to timestep 5.
    Revolve: Checkpoint timestep 5 in memory.
    Revolve: Advance from timestep 5 to timestep 7.
    Revolve: Solve last timestep 7.
    ====== Revolve: Replay from equation 13 (first equation of timestep 7)
             to equation 14 (last equation of timestep 7). ======
    Revolve: Replaying equation 13.
    Revolve: Checkpoint equation 13 in memory.
    Revolve: Replaying equation 14.
    Revolve: Solving adjoint equation 14.
    Revolve: Solving adjoint equation 13.
    Revolve: Delete checkpoint equation 13.
    ...
    ====== Revolve: Replay from equation 2 (first equation of timestep 1)
           to equation 2 (last equation of timestep 1). ======
    Revolve: No need to replay equation 2.
    Revolve: Checkpoint equation 2 in memory.
    Revolve: Solving adjoint equation 2.
    Revolve: Delete checkpoint equation 2.
    ====== Revolve: Replay from equation 0 (first equation of timestep 0)
           to equation 1 (last equation of timestep 0). ======
    Revolve: Replaying equation 0.
    Revolve: No need to replay equation 1.
    Revolve: Solving adjoint equation 1.
    Revolve: Solving adjoint equation 0.
    Revolve: Delete checkpoint equation 0.

There are two categories of checkpointing algorithms: **offline**
algorithms and **online** algorithms.  In the offline case, the number
of timesteps is known in advance, and so the optimal distribution of
checkpoints may be computed a priori (and hence "offline"), while in
the online case, the number of timesteps is not known in advance, and
so the distribution of checkpoints must be computed during the run
itself. Both the offline and online algorithms in revolve only use
disk checkpoints; however, revolve also offers a **multistage**
algorithm, which is a variant of the offline algorithm that uses both
checkpoints in memory and checkpoints on disk.

At present, only the offline and multistage algorithms are
implemented, and so the number of timesteps must be known in
advance. Contributions to interfacing with the online algorithm are
very welcome.

To use checkpointing, the user must specify how many checkpoint slots
are available in memory and on disk. When libadjoint informs
dolfin-adjoint to checkpoint, dolfin-adjoint **records the values of
all variables at that time**. Therefore, *each checkpoint slot is
equivalent to the whole state of memory*. In the above example, both
:py:data:`u` and :py:data:`u_next` will be checkpointed, and so each
checkpoint will store :py:data:`2*V.dim()` floating point
numbers. Keep this in mind when estimating how many checkpoints your
machine can fit.

If you wish to perform large or long computations, you may be
interested in :doc:`running the adjoint in parallel <parallel>`.

.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
