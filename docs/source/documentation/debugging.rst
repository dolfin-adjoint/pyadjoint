.. py:currentmodule:: dolfin_adjoint

=========
Debugging
=========


*******************************************
Visualising the system
*******************************************

It is sometimes useful when debugging a problem to see fenics-adjoint's interpretation of your forward system,
and the other models it derives from that. The :py:meth:`visualise <fenics_adjoint.Tape.visualise>` function visualises the system as a graph.
To do this add

.. code-block:: python

    tape = get_working_tape()
    tape.visualise()

This will show a simple graph, but it is much more readable if we use the DOT format with Graphviz:

.. code-block:: python

    tape = get_working_tape()
    tape.visualise("mygraph", dot = True)

This saves the DOT formatted graph to the file mygraph, which can then be converted to for example a pdf. 
To demonstrate, let us use a simplified version of our old Burgers' equation example:

.. literalinclude:: ../_static/tutorial7.py

Here we solve the equation for only one timestep and save a visualisation in the file simplified_burgers.
                    
|more| Download the `code to find graph`_.

.. _code to find graph: ../_static/tutorial7.py

The resulting `graph`_  is available for inspection.

.. _graph: ../_static/simplified_burgers.pdf

Each node corresponds to an elementary operation, and we see that the structure is what we should expect: four functions and a set of boundary conditions go into an equation
and we get one function out. To increase readability further we can add names to the functions:

.. literalinclude:: ../_static/tutorial7_named.py

The resulting graph is found `here`_.
We see that it is indeed u_next that is found by the equation solve.

.. _here: ../_static/simplified_burgers_named.pdf


In the :doc:`next section
<parallel>` we discuss parallelisation.




.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info

