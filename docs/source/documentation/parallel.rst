========
Parallel
========

Applying algorithmic differentiation tools to parallel source code is still
a major research area, and most adjoint codes that work in parallel manually adjoin the parallel
communication sections of their code.

One of the major advantages of the new high-level abstraction used in dolfin-adjoint is that
the problem of parallelism in adjoint codes simply disappears: indeed, there is not a single
line of parallel-specific code in dolfin-adjoint or libadjoint. For more details on how this
works, see :doc:`the papers <../citing/index>`.

Therefore, **if your forward model runs in parallel, your adjoint will also, with no modification.**
For example, let us take the :doc:`checkpointed adjoint model used in the previous section <checkpointing>`:

.. code-block:: none

    $ mpiexec -n 8 python tutorial5.py
    ...
    Process 0: Convergence orders for Taylor remainder with adjoint information (should all be 2):
      [1.9744066553464978, 1.9872606129796675, 1.9936586367818951, 1.9968385300177882]

.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
