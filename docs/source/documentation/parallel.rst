========
Parallel
========

Applying algorithmic differentiation tools to parallel source code is still
a major research area, and most adjoint codes that work in parallel manually adjoin the parallel
communication sections of their code.

One of the major advantages of the new high-level abstraction used in fenics-adjoint is that
the problem of parallelism in adjoint codes simply disappears: indeed, there is not a single
line of parallel-specific code in fenics-adjoint or pyadjoint. For more details on how this
works, see :doc:`the papers <../citing/index>`.

Therefore, **if your forward model runs in parallel, your adjoint will also, with no modification.**
For example, let us take the adjoint verification program from the section on :doc:`verification <verification>`:

.. code-block:: none

    $ mpiexec -n 4 python tutorial4.py
    ...
    Computed residuals: [8.7896393841476643e-07, 2.2008124728377051e-07, 5.5062931909424556e-08, 1.3771065246938211e-08]
    Computed residuals: [8.7896393841476643e-07, 2.2008124728377051e-07, 5.5062931909424556e-08, 1.3771065246938211e-08]
    Computed residuals: [8.7896393841476643e-07, 2.2008124728377051e-07, 5.5062931909424556e-08, 1.3771065246938211e-08]
    Computed residuals: [8.7896393841476643e-07, 2.2008124728377051e-07, 5.5062931909424556e-08, 1.3771065246938211e-08]
    Computed convergence rates: [1.9977677554998587, 1.9988828855094791, 1.9994412684498588]
    Computed convergence rates: [1.9977677554998587, 1.9988828855094791, 1.9994412684498588]
    Computed convergence rates: [1.9977677554998587, 1.9988828855094791, 1.9994412684498588]
    Computed convergence rates: [1.9977677554998587, 1.9988828855094791, 1.9994412684498588]


.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
