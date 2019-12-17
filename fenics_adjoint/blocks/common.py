import dolfin_adjoint_common.blocks as blocks
from fenics_adjoint.types.compat import Backend

__all__ = ["AssembleBlock", "ProjectBlock", "SolveBlock", "NonlinearSolveBlock"]


class AssembleBlock(blocks.AssembleBlock, Backend):
    pass


class ProjectBlock(blocks.AssembleBlock, Backend):
    pass


class SolveBlock(blocks.SolveBlock, Backend):
    pass


class NonlinearVariationalSolveBlock(blocks.NonlinearVariationalSolveBlock, Backend):
    pass


