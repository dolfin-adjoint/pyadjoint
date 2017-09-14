.. py:currentmodule:: dolfin_adjoint

===================
Miscellaneous notes
===================

**********************************************************************************************************
Naming :py:class:`Functions <dolfin_adjoint.Function>` and :py:class:`Constants <dolfin_adjoint.Constant>`
**********************************************************************************************************

Consider the :doc:`example presented in the tutorial <tutorial>` again:

.. literalinclude:: ../_static/tutorial2.py
    :emphasize-lines: 13,31

Note that the :py:class:`Constant nu <dolfin_adjoint.Constant>` passed to :py:class:`Control(nu) <dolfin_adjoint.Control>` had to be exactly the same variable
as was used in the forward form. This could be quite inconvenient, if the form creation occurs in one file, and the
adjoint is driven from another. To facilitate such cases, it is possible to give the :py:class:`Constant <dolfin_adjoint.Constant>` a :py:data:`name`:

.. code-block:: python

    nu = Constant(0.0001, name="nu")

which may then be used to drive the adjoint. However, the :py:class:`Control <dolfin_adjoint.Control>` class does not have
enough information to determine what kind of control it is: therefore in this case the :py:class:`ConstantControl <dolfin_adjoint.ConstantControl>`
class must be used:

.. code-block:: python

    dJdnu = compute_gradient(J, ConstantControl("nu"))

A full example is given in the following code.

.. literalinclude:: ../_static/tutorial9.py
    :emphasize-lines: 32,36,46

Similarly, it is possible to give :py:class:`Functions <dolfin_adjoint.Function>` names:

.. code-block:: python

    u = Function(ic, name="Velocity")
    u_next = Function(ic, name="VelocityNext")

which can then be used in other places with :py:class:`FunctionControl <dolfin_adjoint.FunctionControl>`:

.. code-block:: python

    dJdic = compute_gradient(J, FunctionControl("Velocity"))

A full example is given in the following code.

.. literalinclude:: ../_static/tutorial10.py
    :emphasize-lines: 12,13,36,46

**********************
Lower-level interfaces
**********************

A lower-level interface is available to loop over the adjoint solutions (backwards in time) for some manual calculation.
This uses the :py:func:`compute_adjoint <dolfin_adjoint.compute_adjoint>` function:

.. code-block:: python

    J = Functional(inner(u, u)*dx*dt[FINISH_TIME])
    for (variable, solution) in compute_adjoint(J):
        ...

Here, :py:data:`solution` is a :py:class:`dolfin.Function <dolfin_adjoint.Function>` storing a particular adjoint solution,,
and :py:data:`variable` is a :py:class:`libadjoint.Variable <libadjoint.Variable>` that describes
which forward solution it corresponds to. 

Similarly, it is possible to loop over the tangent linear solutions with the :py:func:`compute_tlm <dolfin_adjoint.compute_tlm>` function:

.. code-block:: python

    param = Control(nu)
    for (variable, solution) in compute_tlm(param):
        ...


