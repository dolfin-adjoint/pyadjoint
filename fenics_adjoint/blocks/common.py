import dolfin_adjoint_common.blocks as blocks
from fenics_adjoint.compat import Backend

__all__ = ["AssembleBlock", "ProjectBlock", "SolveBlock", "NonlinearVariationalSolveBlock", "FunctionAssignBlock"]


class AssembleBlock(blocks.AssembleBlock, Backend):
    pass


class ProjectBlock(blocks.ProjectBlock, Backend):
    pass


class SolveBlock(blocks.SolveBlock, Backend):
    pass


class NonlinearVariationalSolveBlock(blocks.NonlinearVariationalSolveBlock, Backend):
    pass


class FunctionAssignBlock(blocks.FunctionAssignBlock, Backend):
    pass



