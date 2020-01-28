import dolfin_adjoint_common.blocks as blocks
from fenics_adjoint.compat import Backend

__all__ = ["AssembleBlock", "GenericSolveBlock",
           "FunctionAssignBlock", "DirichletBCBlock",
           "ConstantAssignBlock"]


class AssembleBlock(blocks.AssembleBlock, Backend):
    pass


class GenericSolveBlock(blocks.GenericSolveBlock, Backend):
    pass


class FunctionAssignBlock(blocks.FunctionAssignBlock, Backend):
    pass


class DirichletBCBlock(blocks.DirichletBCBlock, Backend):
    pass


class ConstantAssignBlock(blocks.ConstantAssignBlock, Backend):
    pass
