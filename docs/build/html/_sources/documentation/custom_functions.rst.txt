.. py:currentmodule:: fenics_adjoint

=======================
Adding Custom Functions
=======================

As mentioned in the :doc:`first section  <tutorial>` of this tutorial fenics-adjoint
works by overloading parts of FEniCS so that it may build up an annotation by recording
each step of the forward model. The list of overloaded functions and objects is found
in the :doc:`API reference <api>`. The part of fenics-adjoint that takes care of the fundamental
annotation is pyadjoint, which is independent of FEniCS.
fenics-adjoint tells pyadjoint how to handle FEniCS types and functions.
If the forward model uses custom functions rather than the
standard FEniCS functions, pyadjoint won't know how to record
these steps, therefore we have to tell it how, by overloading the functions ourselves.

****************
A Simple Example
****************

Suppose we have a module we want to use with our FEniCS model,
in this example the module will be named :py:data:`normalise` and consist of
only one function: :py:data:`normalise(func)`. The module looks like this:

.. literalinclude:: ../_static/overloading/normalise.py

|more| `Download this file`_

.. _`Download this file`: ../_static/overloading/normalise.py

The function :py:data:`normalise(func)` normalises the vector form of a FEniCS function,
then returns the FEniCS function form of that normalised vector. A simple fenics
program that uses this function might look like this:

.. literalinclude:: ../_static/overloading/tutorial9.py

|more| `Download this example`_

.. _`Download this example`: ../_static/overloading/tutorial9.py

Here we define a function on a space, normalise it with our function and integrate it
over the space. Now we want to know the gradient of :math:`J` with respect to the initial
conditions, we could try simply adding

.. code-block:: python

   from fenics_adjoint import *

and

.. code-block:: python

   dJdf = compute_gradient(J,f)

but that won't work, because pyadjoint does not know that it should record
the normalisation and it does not know what the derivative of the normalisation is.
We should create a new module that overloads :py:data:`normalise(func)`, telling
pyadjoint how to deal with it.

**********************
Overloading a function
**********************

Let us now create a module overloading the :py:data:`normalise(func)` function.
We need to start by importing the FEniCS and fenics-adjoint modules, along with
some specific functions needed for overloading and of course the function we want to
overload.

.. code-block:: python

   from fenics import *
   from fenics_adjoint import *

   from pyadjoint import Block, annotate_tape, stop_annotating
   from fenics_adjoint.types import create_overloaded_object

   from normalise import normalise

------------------------
The overloading function
------------------------

Since we are overloading :py:data:`normalise(func)` we need to change it's name
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

- We check whether or not we should be annotating. If the user passes

  .. code-block:: python

     annotate_tape = False

  as a keyword argument we should treat the function call exactly as if we were just
  using the non-overloaded version of  :py:data:`normalise(func)`.

- If we are annotating we get the current tape, make a block, which
  are the building blocks of the tape and then add the new block to the tape.
  :py:data:`NormaliseBlock(func)` is the constructor of the class
  :py:class:`NormaliseBlock(Block)`, which we will implement and which contains the
  information about how pyadjoint should handle our function.

- We compute the normalisation with our non-overloaded function, and then make sure
  that the output is an overloaded object that can be properly handled by pyadjoint.

- If we are annotating we add the output to our block.

- And finally we return the output.

This is quite general, the only things that specifically refers to normalisation are
:py:data:`backend_normalise(func)` and :py:data:`NormaliseBlock(func)`, and the
overloading function will look very similar to this in most cases.

---------------
The Block class
---------------

The class :py:class:`NormaliseBlock(Block)` is a subclass of
:py:class:`Block <pyadjoint.Block>` from the pyadjoint module. In addition to
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
Then we tell pyadjoint that the operation this block represents depends
on the function :py:data:`func`. As :py:data:`func` should be an overloaded object it has a
:py:meth:`get_block_output` method.

Next we can define a :py:meth:`__str__` method. This gives a name to the block,
so the output of this is for example how the block is represented in graphs made
with :py:meth:`visualise <pyadjoint.Tape.visualise>` as explained in the section
on :doc:`debugging <debugging>`.

.. code-block:: python

   def __str__(self):
       return "NormaliseBlock"

We need a :py:meth:`recompute <pyadjoint.Block.recompute>` method that can
recompute the block.

.. code-block:: python

   def recompute(self):
       dependencies = self.get_dependencies()
       func = dependencies[0].get_saved_output()
       output = backend_normalise(func, **self.kwargs)
       self.get_outputs()[0].checkpoint = output

We get the inputs from the dependencies, calculate the function and save it to the ouput.

-----------
The adjoint
-----------


The method :py:meth:`evaluate_adj` should evaluate the adjoint gradient of the block.
In the :doc:`mathematical background <maths/index>` we discussed the tangent linear model
and the adjoint on the level of the whole model. Here we consider more concretely
how fenics-adjoint treats each block. pyadjoint treats a forward model as a series of equation solves.
Some of these equations are complicated PDEs that are solved by the FEniCS function :py:func:`solve <fenics.solve>`,
but others are of the straightforward form

.. math:: y = f(x_1,\ldots,x_n),

where :math:`y` is the only unknown. Our :py:data:`normalise` function may be represented by this kind of equation.
When differentiating a functional pyadjoint works by considering each block as a link in chain formed by
the chain rule. If a functional is the result of a series of straightforward transformations on an initial condition:

.. math:: J(u_n(u_{n-1}(\ldots(u_0)\ldots))),

then by the chain rule

.. math::

   \frac{\mathrm{d}J}{\mathrm{d}u_0} = \frac{\partial J}{\partial u_n}\frac{\partial u_n}{\partial u_{n-1}}\ldots\frac{\partial u_1}{\partial u_0}.

If we consider instead the adjoint model we will find the transpose of :math:`\frac{\mathrm{d}J}{\mathrm{d}u_0}`:

.. math::

   \frac{\mathrm{d}J}{\mathrm{d}u_0}^* = \frac{\partial u_1}{\partial u_0}^*\frac{\partial u_n}{\partial u_{n-1}}^*\ldots\frac{\partial J}{\partial u_n}^*.

Calculating from the right we find that for each link

.. math::

   y_i = \frac{\partial u_i}{\partial u_{i+1}}^*y_{i+1},

where

.. math::

   y_{n+1} = \frac{\partial J}{\partial u_n}.

and

.. math::

   y_0 = \frac{\mathrm{d} J}{\mathrm{d} u_0}

Each block only needs to find the transpose of its own gradient!
This is implemented in :py:meth:`evaluate_adj`.

-------------------
Back to our example
-------------------

Mathematically our normalisation block may be represented in index notation as

.. math::

   f(x_i) = \frac{x_i}{||x||}.

The gradient matrix is

.. math::

   \frac{\partial f(x_i)}{\partial x_j} = \frac{1}{||x||} \delta_{ij} - \frac{x_i x_j}{||x||^3}

:py:meth:`evaluate_adj` takes a vector as input and returns that vector
multiplied with the transpose of the gradient:

.. math:: \nabla f^* \cdot y = \sum_j \frac{\partial f(x_j)}{\partial x_i} y_j =
          \sum_{j} \frac{1}{||x||} \delta_{ij} y_j -
          \frac{x_i x_j}{||x||^3} y_j = \frac{y_i}{||x||} - \frac{x_i}{||x||^3} \sum_j x_j y_j

Now let us look at the implementation:

.. code-block:: python

   def evaluate_adj(self):
       adj_input = self.get_outputs()[0].adj_value
       dependency = self.get_dependencies()[0]
       x = dependency.get_saved_output().vector()


:py:data:`adj_input` is the vector :math:`y` above. As we are going *backwards* through the forward model
it is the output of :py:meth:`evaluate_adj` for the *output* of our normalisation
block. Then we get the value of the input to our block and save it as a vector.
Next, we should compute the value:

.. code-block:: python

       adj_output = x.copy()

       xnorm = x.norm('l2')

       const = 0
       for i in range(len(x)):
           const += adj_input[i][0]*x[i][0]
       const /= xnorm**3

       for i in range(len(x)):
           adj_output[i] = adj_input[i][0]/xnorm - const*x[i][0]

Finally we save :py:data:`adj_output` so that it may be propagated up the chain

.. code-block:: python

       dependency.add_adj_output(adj_output)

|more| `download the overloaded module`_

.. _`download the overloaded module`: ../_static/overloading/normalise_overloaded.py

That's it! Now we are ready to use our function :py:data:`normalise` with fenics-adjoint.
Let us perform a taylor test to see if it works:

.. literalinclude:: ../_static/overloading/tutorial9_overloading.py

|more| `download this test`_

.. _`download this test`: ../_static/overloading/tutorial9_overloading.py

This gives the output:

.. code-block:: none

   Computed residuals: [5.719808547972123e-06, 1.4356712128879936e-06, 3.5963468743448646e-07, 8.999840626988198e-08]
   Computed convergence rates: [1.9942414669485427, 1.997121308032896, 1.9985608192606437]

It works.




.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
