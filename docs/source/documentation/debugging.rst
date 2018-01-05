.. py:currentmodule:: dolfin_adjoint

=========
Debugging
=========


*******************************************
Visualising the system
*******************************************

It is sometimes useful when debugging a problem to see dolfin-adjoint's interpretation of your forward system,
and the other models it derives from that. The :py:meth:`visualise <pyadjoint.Tape.visualise>` function visualises the system as a graph.
To do this add

.. code-block:: python

    tape = get_working_tape()
    tape.visualise()

This will show a simple graph, but it is much more readable if we use the DOT format with Graphviz:

.. code-block:: python

    tape = get_working_tape()
    tape.visualise("mygraph", dot = True)

This saves the DOT formatted graph to the file mygraph, which can then be converted to for example a png file.
To demonstrate, let us use a simplified version of our old Burgers' equation example:

.. literalinclude:: ../_static/tutorial7.py

Here we solve the equation for only one timestep and save a visualisation in the file simplified_burgers.

|more| Download the `code to find graph`_.

.. _code to find graph: ../_static/tutorial7.py

Compiling the dot file results in the following graph:

.. image:: ../_static/simplified_burgers.png

Each node corresponds to an elementary operation, and we see that the structure is what we should expect: four functions and a set of boundary conditions go into an equation
and we get one function out. To increase readability further we can add names to the functions:

.. literalinclude:: ../_static/tutorial7_named.py

The resulting graph is the following:

.. image:: ../_static/simplified_burgers_named.png

We see that it is indeed :py:data:`u_next` that is found by the equation solve.


|more| Download the `code to find the graph with names`_.

.. _code to find the graph with names: ../_static/tutorial7_named.py

In the :doc:`next section
<parallel>` we discuss parallelisation.




.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
