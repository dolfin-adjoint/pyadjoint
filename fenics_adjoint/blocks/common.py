import dolfin_adjoint_common.blocks as blocks
from fenics_adjoint.compat import Backend

__all__ = ["AssembleBlock", "ProjectBlock", "GenericSolveBlock",
           "SolveLinearSystemBlock", "SolveVarFormBlock",
           "NonlinearVariationalSolveBlock", "FunctionAssignBlock",
           "DirichletBCBlock"]


class AssembleBlock(blocks.AssembleBlock, Backend):
    pass


class ProjectBlock(blocks.ProjectBlock, Backend):
    pass


class GenericSolveBlock(blocks.GenericSolveBlock, Backend):
    pass


class SolveLinearSystemBlock(blocks.SolveLinearSystemBlock, Backend):
    pass


class SolveVarFormBlock(blocks.SolveVarFormBlock, Backend):
    pass


class NonlinearVariationalSolveBlock(blocks.NonlinearVariationalSolveBlock, Backend):
    pass


class FunctionAssignBlock(blocks.FunctionAssignBlock, Backend):
    pass


class DirichletBCBlock(blocks.DirichletBCBlock, Backend):
    pass

