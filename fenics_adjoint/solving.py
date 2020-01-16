import backend
import ufl

from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape
from .blocks import SolveVarFormBlock, SolveLinearSystemBlock

# Type dependencies

# TODO: Clean up: some inaccurate comments. Reused code. Confusing naming with dFdm when denoting the control as c.


def solve(*args, **kwargs):
    """This solve routine wraps the real Dolfin solve call. Its purpose is to annotate the model,
    recording what solves occur and what forms are involved, so that the adjoint and tangent linear models may be
    constructed automatically by pyadjoint.

    To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
    Dolfin solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
    for the purposes of the adjoint computation (such as projecting fields to other function spaces
    for the purposes of visualisation).

    The overloaded solve takes optional callback functions to extract adjoint solutions.
    All of the callback functions follow the same signature, taking a single argument of type Function.

    Keyword Args:
        adj_cb (function, optional): callback function supplying the adjoint solution in the interior.
            The boundary values are zero.
        adj_bdy_cb (function, optional): callback function supplying the adjoint solution on the boundary.
            The interior values are not guaranteed to be zero.
        adj2_cb (function, optional): callback function supplying the second-order adjoint solution in the interior.
            The boundary values are zero.
        adj2_bdy_cb (function, optional): callback function supplying the second-order adjoint solution on
            the boundary. The interior values are not guaranteed to be zero.

    """
    annotate = annotate_tape(kwargs)
    if annotate:
        tape = get_working_tape()

        solve_block_type = SolveVarFormBlock
        if not isinstance(args[0], ufl.equation.Equation):
            solve_block_type = SolveLinearSystemBlock

        sb_kwargs = solve_block_type.pop_kwargs(kwargs)
        sb_kwargs.update(kwargs)
        block = solve_block_type(*args, **sb_kwargs)
        tape.add_block(block)

    with stop_annotating():
        output = backend.solve(*args, **kwargs)

    if annotate:
        if hasattr(args[1], "create_block_variable"):
            block_variable = args[1].create_block_variable()
        else:
            block_variable = args[1].function.create_block_variable()
        block.add_output(block_variable)

    return output
