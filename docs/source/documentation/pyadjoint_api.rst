:orphan:

.. _pyadjoint-api-reference:

============================
pyadjoint API reference
============================

See also the :doc:`dolfin-adjoint API reference <api>`.

.. automodule:: pyadjoint

************
Core classes
************

.. autoclass:: Tape

    .. automethod:: add_block
    .. automethod:: visualise
    .. autoproperty:: progress_bar

.. autoclass:: Block

    .. automethod:: pop_kwargs
    .. automethod:: add_dependency
    .. automethod:: add_output
    .. automethod:: evaluate_adj
    .. automethod:: prepare_evaluate_adj
    .. automethod:: evaluate_adj_component
    .. automethod:: evaluate_tlm
    .. automethod:: prepare_evaluate_tlm
    .. automethod:: evaluate_tlm_component
    .. automethod:: evaluate_hessian
    .. automethod:: prepare_evaluate_hessian
    .. automethod:: evaluate_hessian_component
    .. automethod:: recompute
    .. automethod:: prepare_recompute_component
    .. automethod:: recompute_component

.. autoclass:: pyadjoint.block_variable.BlockVariable

.. autoclass:: OverloadedType

    .. automethod:: _ad_init_object
    .. automethod:: _ad_convert_type
    .. automethod:: _ad_create_checkpoint
    .. automethod:: _ad_restore_at_checkpoint
    .. automethod:: _ad_mul
    .. automethod:: _ad_imul
    .. automethod:: _ad_add
    .. automethod:: _ad_iadd
    .. automethod:: _ad_dot
    .. automethod:: _ad_assign_numpy
    .. automethod:: _ad_to_list
    .. automethod:: _ad_copy
    .. automethod:: _ad_dim

**********************
Core utility functions
**********************

.. autofunction:: get_working_tape
.. autofunction:: set_working_tape
.. autofunction:: pyadjoint.tape.no_annotations
.. autoclass:: stop_annotating
.. autofunction:: annotate_tape
.. autofunction:: pyadjoint.overloaded_type.create_overloaded_object
.. autofunction:: pyadjoint.overloaded_type.register_overloaded_type

**************
User interface
**************

.. autoclass:: Control
.. autofunction:: compute_gradient
.. autofunction:: compute_hessian
.. autoclass:: pyadjoint.placeholder.Placeholder
.. autoclass:: ReducedFunctional

    .. automethod:: __call__
    .. automethod:: derivative
    .. automethod:: hessian
    .. automethod:: optimize_tape

.. autoclass:: pyadjoint.reduced_functional_numpy.ReducedFunctionalNumPy
.. autofunction:: taylor_test


******************
Overloaded objects
******************

.. autoclass:: AdjFloat
