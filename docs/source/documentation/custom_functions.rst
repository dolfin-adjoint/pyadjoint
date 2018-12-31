.. py:currentmodule:: fenics_adjoint

=======================
Adding Custom Functions
=======================

As mentioned in the :doc:`first section  <tutorial>` of this tutorial dolfin-adjoint
works by overloading parts of FEniCS so that it may build up an annotation by recording
each step of the forward model. The list of overloaded functions and objects is found
in the :doc:`API reference <api>`. The part of dolfin-adjoint that takes care of the fundamental
annotation is pyadjoint, which is independent of FEniCS.
dolfin-adjoint tells pyadjoint how to handle FEniCS types and functions.
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

   dJdf = compute_gradient(J, Control(f))

but that won't work, because pyadjoint does not know that it should record
the normalisation and it does not know what the derivative of the normalisation is.
We should create a new module that overloads :py:data:`normalise(func)`, telling
pyadjoint how to deal with it.

**********************
Overloading a function
**********************

Let us now create a module overloading the :py:data:`normalise(func)` function.
We need to start by importing the FEniCS and dolfin-adjoint modules, along with
some specific functions needed for overloading and of course the function we want to
overload.

.. code-block:: python

   from fenics import *
   from fenics_adjoint import *

   from pyadjoint import Block

   from normalise import normalise

Since we are overloading :py:data:`normalise(func)` we need to change it's name
to keep access to it:

.. code-block:: python

   backend_normalise = normalise

---------------
The Block class
---------------

The pyadjoint tape consists of instances of :py:class:`Block <pyadjoint.Block>` subclasses.
These subclasses implement methods that can compute partial derivatives of their respective function.
Thus, to properly overload :py:data:`normalise` we must implement a :py:class:`Block <pyadjoint.Block>` subclass,
which we call :py:class:`NormaliseBlock`.
In addition to writing a constructor we have to override the methods :py:meth:`evaluate_adj_component` and
:py:meth:`recompute_component`, we will also add a :py:meth:`__str__` method.
In our example the constructor may look like this

.. code-block:: python

   class NormaliseBlock(Block):
      def __init__(self, func, **kwargs):
          super(NormaliseBlock, self).__init__()
          self.kwargs = kwargs
          self.add_dependency(func.block_variable)

We call the superclass constructor and  save the key word arguments.
Then we tell pyadjoint that the operation this block represents depends
on the function :py:data:`func`. As :py:data:`func` should be an overloaded object it has a
:py:meth:`block_variable` attribute.

Next we can define a :py:meth:`__str__` method. This gives a name to the block,
so the output of this is for example how the block is represented in graphs made
with :py:meth:`visualise <pyadjoint.Tape.visualise>` as explained in the section
on :doc:`debugging <debugging>`.

.. code-block:: python

   def __str__(self):
       return "NormaliseBlock"

We need a :py:meth:`recompute <pyadjoint.Block.recompute_component>` method that can
recompute the function with new inputs.

.. code-block:: python

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_normalise(inputs[0])

We get a list of new inputs which is of length 1 because we only have one input variable.
Or more precisely, we only added one dependency in the constructor.

-----------
The adjoint
-----------


The method :py:meth:`evaluate_adj_component` should evaluate the one component of the vector-Jacobian product.
In the :doc:`mathematical background <maths/index>` we discussed the tangent linear model
and the adjoint on the level of the whole model. Here we consider more concretely
how dolfin-adjoint treats each block. pyadjoint treats a forward model as a series of equation solves.
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

   \frac{\mathrm{d}J}{\mathrm{d}u_0}^* = \frac{\partial u_1}{\partial u_0}^*\frac{\partial u_2}{\partial u_{1}}^*\ldots\frac{\partial J}{\partial u_n}^*.

Calculating from the right we find that for each link

.. math::

   y_i = \frac{\partial u_i}{\partial u_{i-1}}^*y_{i+1},

where

.. math::

   y_{n+1} = \frac{\partial J}{\partial u_n}^*.

and

.. math::

   y_1 = \frac{\mathrm{d} J}{\mathrm{d} u_0}^*

Each block only needs to find the transpose of its own gradient!
This is implemented in :py:meth:`evaluate_adj`.

-------------------
Back to our example
-------------------

Mathematically our normalisation block may be represented in index notation as

.. math::

   f(x_i) = \frac{x_i}{||x||}.

The Jacobian matrix consists of the entries

.. math::

   \frac{\partial f(x_i)}{\partial x_j} = \frac{1}{||x||} \delta_{ij} - \frac{x_i x_j}{||x||^3}

:py:meth:`evaluate_adj` takes a vector as input and returns the transpose of the Jacobian matrix
 multiplied with that vector:

.. math:: \nabla f^* \cdot y = \sum_j \frac{\partial f(x_j)}{\partial x_i} y_j =
          \sum_{j} \frac{1}{||x||} \delta_{ij} y_j -
          \frac{x_i x_j}{||x||^3} y_j = \frac{y_i}{||x||} - \frac{x_i}{||x||^3} \sum_j x_j y_j

:py:meth:`evaluate_adj_component` works as :py:meth:`evaluate_adj`,
but computes only the component that corresponds to a single dependency (input).
In other words, given an index :math:`i` :py:meth:`evaluate_adj_component` computes
the component :math:`\left(\nabla f^* \cdot y\right)_i`.

By default, :py:meth:`evaluate_adj` calls :py:meth:`evaluate_adj_component` for each of the relevant components.

Now let us look at the implementation:

.. code-block:: python

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        x = inputs[idx].vector()
        inv_xnorm = 1.0 / x.norm('l2')
        return inv_xnorm * adj_input - inv_xnorm ** 3 * x.inner(adj_input) * x

:py:meth:`evaluate_adj_component` takes 5 arguments:

- :py:data:`inputs` is a list of the inputs where we compute the derivative, i.e :math:`x` in the above derivations.
  This list has the same length as the list of dependencies.
- :py:data:`adj_inputs` is a list of the adjoint inputs, i.e :math:`y_{i+1}` above with this method representing the computation of :math:`y_i`.
  This list has the same length as the list of outputs.
- :py:data:`block_variable` is the block variable of the dependency (input) that we differentiate with respect to.
- :py:data:`idx` is the index of the dependency, that we differentiate with respect to, in the list of dependencies.
  Given a function output :math:`z = f(x, y)`, where the dependency list is :py:data:`[x, y]`, then :math:`(\partial z/\partial x)^*`
  for :py:data:`idx == 0` and :math:`(\partial z/\partial y)^*` for :py:data:`idx == 1`.
- :py:data:`prepared` can be anything. It is the return value of :py:meth:`prepare_evaluate_adj`,
  which is run before :py:meth:`evaluate_adj_component` is called for each relevant dependency
  and the default return value is :py:data:`None`.
  If your implementation would benefit from doing some computations independent of the relevant dependencies,
  you should consider implementing :py:meth:`prepare_evaluate_adj`.
  For example, for :py:meth:`solve` the adjoint equation is solved in :py:meth:`prepare_evaluate_adj`,
  and the adjoint solution is provided in the :py:data:`prepared` parameter.

For more in-depth documentation on Blocks in pyadjoint, see

------------------------
The overloading function
------------------------

Now we are ready to define our overloaded function.
For simple functions, where the function return value is the output,
pyadjoint offers a convenience function for overloading.
For this example, we utilize this convenience function:

.. code-block:: python

    from pyadjoint.overloaded_function import overload_function
    normalise = overload_function(normalise, NormaliseBlock)

|more| `download the overloaded module`_

.. _`download the overloaded module`: ../_static/overloading/normalise_overloaded.py

That's it! Now we are ready to use our function :py:data:`normalise` with dolfin-adjoint.
Let us perform a taylor test to see if it works:

.. literalinclude:: ../_static/overloading/tutorial9_overloading.py

|more| `download this test`_

.. _`download this test`: ../_static/overloading/tutorial9_overloading.py

This gives the output:

.. code-block:: none

   Running Taylor test
   Computed residuals: [5.719808547999933e-06, 1.4356712128880207e-06, 3.596346874345e-07, 8.999840626988876e-08]
   Computed convergence rates: [1.99424146695553, 1.9971213080328687, 1.9985608192605893]

It works.




.. |more| image:: ../_static/more.png
          :align: middle
          :alt: more info
