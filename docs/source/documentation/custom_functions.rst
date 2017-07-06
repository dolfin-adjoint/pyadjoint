.. py:currentmodule:: fenics_adjoint

=======================
Adding Custom Functions
=======================

As mentioned in the :doc:`first section of this tutorial <tutorial>` fenics adjoint
works by overloading parts of fenics so that it may build up an annotation by recording
each step of the forward model. The list of overloaded functions and objects is found
in the :doc:`api <api>`. If the forward model uses custom functions rather than the
standard fenics functions, fenics_adjoint won't automatically know how to record
these steps, therefore we have to tell it how, by overloading the function ourselves.

****************
A Simple Example
****************

Suppose we have a module we want to use with our fenics model,
in this example the module will be named :py:data:`normalise` and consist of
only one function: :py:func:`normalise(func)`. The module looks like this:

.. literalinclude:: ../_static/overloading/normalise.py

|more| `Download this file`_

.. _`Download this file`: ../_static/overloading/normalise.py

The function :py:func:`normalise(func)` normalises the vector form of a fenics function,
then returns a the fenics function form of that normalised vector. A simple fenics
program that uses this function might look like this:

.. literalinclude:: ../_static/overloading/tutorial9.py

|more| `Download this example`_

.. _`Download this example`: ../_static/overloading/normalise.py

Here we define a function on a space, normalise it with our function and integrate it
over the space. Now we want to know the gradient of :math:`J` with respect to the initial
conditions, we could try simply adding

.. code-block:: python

   from fenics_adjoint import *

and

.. code-block:: python

   dJdf = compute_gradient(J,f)

but that won't work, because fenics_adjoint does not know that it should record
the normalisation and it does not know what the derivative of the normalisation is.
We should create a new module that overloads :py:func:`normalise(func)`, telling
fenics adjoint how to deal with it.

**********************
Overloading a function
**********************

Let us now create a module overloading the :py:func:`normalise(func)` function.
We need to start by importing the fenics and fenics_adjoint modules, along with
some specific functions needed for overloading and of course the function we want to
overload.

.. code-block:: python

   from fenics import *
   from fenics_adjoint import *

   from pyadjoint.block import Block
   from pyadjoint.tape import annotate_tape, stop_annotating
   from fenics_adjoint.types import create_overloaded_object

   from normalise import normalise

------------
The function
------------

Since we are overloading :py:func:`normalise(func)` we need to change it's name
to keep acces to it:

.. code-block:: python

   backend_normalise = normalise

Now we are ready to write the overloaded function:

.. code-block:: python

   def normalise(func, **kwargs):
    annotate = annotate_tape(kwargs)

    if annotate:
        tape = get_working_tape()
        block = NormaliseBlock(func)
        tape.add_block(block)

    with stop_annotating():
        output = backend_normalise(func, **kwargs)

    output = create_overloaded_object(output)

    if annotate:
        block.add_output(output.create_block_output())

    return output

So what is going on here:

- We check whether or not we should be annotating, if the user passes

  .. code-block:: python

     annotate_tape = False

  as a keyword argument we should treat the function call exactly as if we were just
  using the non-overloaded version of  :py:func:`normalise(func)`.

- If we are annotating we get the tape were we are annotating, make a block, which
  are the building blocks of the tape and then add the new block to the tape.
  :py:meth:`NormaliseBlock(func)` is the constructor of the class
  :py:class:`NormaliseBlock(Block)`, which we will implement and which contains the
  information about how fenics_adjoint should handle our function.

- We compute the normalisation with our non-overloaded function, and then make sure
  that the output is an overloaded object that can be properly handled by fenics_adjoint.

- If we are annotating we add the output to our block.

- And finally we return the output.

This is quite general, the only things that specifically refers to normalisation are
:py:func:`backend_normalise(func)` and :py:meth:`NormaliseBlock(func)`, and the
overloading function will look very similar to this in most cases.

---------------
The Block class
---------------

The class :py:class:`NormaliseBlock(Block)` is a subclass of
:py:class:`Block <pyadjoint.block.Block>` from the pyadjoint module. In addition to
writing a constructor we have to override the methods :py:meth:`evaluate_adj` and
:py:meth:`recompute`, we will also add a :py:meth:`__str__` method.
In our example the constructor may look like this

.. code-block:: python

   class NormaliseBlock(Block):
      def __init__(self, func, **kwargs):
          super(NormaliseBlock, self).__init__()
          self.kwargs = kwargs
          self.add_dependency(func.get_block_output())

We call the superclass constructor and  save the key word arguments.
Then we tell fenics_adjoint that the operation this block represents depends
on the function :py:data:`func`. As :py:data:`func` is an overloaded object it has a
:py:meth:`get_block_output` method.

Next we can define a :py:meth:`__str__` method. This gives a name to the block,
so the output of this is for example how the block is represented in graphs made
with :py:meth:`visualise <pyadjoint.tape.Tape.visualise>` as explained in the section
on :doc:`debugging <debugging>`.

.. code-block:: python

   def __str__(self):
       return "NormaliseBlock"

-----------
The adjoint
-----------

The method :py:meth:`evaluate_adj` is should evaluate the adjoint of the block.
In the :doc:`mathematical background <maths/index>` we discussed the tangent linear model
and the adjoint. We saw that the .........

If we then consider our system as a series of linear equations:

.. math:: y_i = A_i y_{i-1}

with initial condition

.. math:: y_0 = x

we can write our system as a block matrix system like so:

.. math:: \begin{pmatrix}
          I&&&&&\\
          -A_1&I&&&&\\
          &&\ddots&&&\\
          &&-A_i&I&&\\
          &&&&\ddots&\\
          &&&&-A_n&I\\
          \end{pmatrix}
          \begin{pmatrix}
          y_0\\
          \vdots\\
          \\
          y_i\\
          \vdots\\
          y_n\\
          \end{pmatrix}
          =
          \begin{pmatrix}
          x\\
          0\\
          \vdots\\
          \\
          \\
          0\\
          \end{pmatrix},

where the empty spaces not indicated by dots are 0.
That the equations may be solved in sequence is here manifested in the matrix being lower triangular.
Let us use bold typefaces to indicate this large system

.. math::

   \bf{Ay} = \bf{x}

We note that the tangent linear model is now

.. math::

   \left(\bf{A} + \frac{\partial \bf{A}}{\partial \bf{y}}\bf{y}\right)\frac{\mathrm{d}\bf{y}}{\mathrm{d} x} = -\frac{\partial\bf{x}}{\partial x}

and so the adjoint model is

.. math::

   \left(\bf{A} + \frac{\partial \bf{A}}{\partial \bf{y}}\bf{y}\right)^*\lambda = \frac{\partial J}{\partial \bf{y}}

We note that our matrix is now *upper* triangular: we may solve for lambda from the end and backwards!
If the functional we are interested in only explicitly depends on the final value for :math:`y` we are now in a similar situation to
the forward model......
Mathematically our block may be represented in index notation as

.. math::

   f(x_i) = \frac{x_i}{||x||}.

The derivative matrix is

.. math::

   \frac{\partial f(x_i)}{\partial x_j} = \frac{1}{||x||} \delta_{ij} - \frac{x_i x_j}{||x||^3}

and thus the adjoint is

.. math::

   \left(\frac{\partial J}{\partial x_i}\right)^* = \sum_{j} \frac{\partial f(x_j)}{\partial x_i}y_j
   = \sum_{j} \frac{1}{||x||} \delta_{ij} y_j -
   \frac{x_i x_j}{||x||^3} y_j = \frac{y_i}{||x||} - \frac{x_i}{||x||^3} \sum_j x_j y_j







.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
